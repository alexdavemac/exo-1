import AppKit
import Combine
import Foundation

private let customNamespaceKey = "EXOCustomNamespace"
private let hfTokenKey = "EXOHFToken"
private let enableImageModelsKey = "EXOEnableImageModels"
private let apiPortKey = "EXOAPIPort"

@MainActor
final class ExoProcessController: ObservableObject {
    enum Status: Equatable {
        case stopped
        case starting
        case running
        case monitoring  // Connected to externally-managed exo instance
        case failed(message: String)

        var displayText: String {
            switch self {
            case .stopped:
                return "Stopped"
            case .starting:
                return "Starting…"
            case .running:
                return "Running"
            case .monitoring:
                return "Monitoring"
            case .failed:
                return "Failed"
            }
        }

        var isConnected: Bool {
            switch self {
            case .running, .monitoring:
                return true
            default:
                return false
            }
        }
    }

    static let exoDirectoryURL: URL = {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".exo")
    }()

    @Published private(set) var status: Status = .stopped
    @Published private(set) var isMonitoringExternal: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var launchCountdownSeconds: Int?
    @Published var customNamespace: String = {
        return UserDefaults.standard.string(forKey: customNamespaceKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(customNamespace, forKey: customNamespaceKey)
        }
    }
    @Published var hfToken: String = {
        return UserDefaults.standard.string(forKey: hfTokenKey) ?? ""
    }()
    {
        didSet {
            UserDefaults.standard.set(hfToken, forKey: hfTokenKey)
        }
    }
    @Published var enableImageModels: Bool = {
        return UserDefaults.standard.bool(forKey: enableImageModelsKey)
    }()
    {
        didSet {
            UserDefaults.standard.set(enableImageModels, forKey: enableImageModelsKey)
        }
    }
    @Published var apiPort: Int = {
        let saved = UserDefaults.standard.integer(forKey: apiPortKey)
        return saved > 0 ? saved : 52415
    }()
    {
        didSet {
            UserDefaults.standard.set(apiPort, forKey: apiPortKey)
        }
    }

    var apiBaseURL: URL {
        URL(string: "http://127.0.0.1:\(apiPort)")!
    }

    private var process: Process?
    private var runtimeDirectoryURL: URL?
    private var pendingLaunchTask: Task<Void, Never>?

    func launchIfNeeded() {
        guard process?.isRunning != true else { return }
        launch()
    }

    func launch() {
        do {
            guard process?.isRunning != true else { return }
            cancelPendingLaunch()
            status = .starting
            lastError = nil
            let runtimeURL = try resolveRuntimeDirectory()
            runtimeDirectoryURL = runtimeURL

            let executableURL = runtimeURL.appendingPathComponent("exo")

            let child = Process()
            child.executableURL = executableURL
            let exoHomeURL = Self.exoDirectoryURL
            try? FileManager.default.createDirectory(
                at: exoHomeURL, withIntermediateDirectories: true
            )
            child.currentDirectoryURL = exoHomeURL
            child.environment = makeEnvironment(for: runtimeURL)

            child.standardOutput = FileHandle.nullDevice
            child.standardError = FileHandle.nullDevice

            child.terminationHandler = { [weak self] proc in
                Task { @MainActor in
                    guard let self else { return }
                    self.process = nil
                    switch self.status {
                    case .stopped:
                        break
                    case .failed:
                        break
                    default:
                        self.status = .failed(
                            message: "Exited with code \(proc.terminationStatus)"
                        )
                        self.lastError = "Process exited with code \(proc.terminationStatus)"
                    }
                }
            }

            try child.run()
            process = child
            status = .running
        } catch {
            process = nil
            status = .failed(message: "Launch error")
            lastError = error.localizedDescription
        }
    }

    func stop() {
        guard let process else {
            status = .stopped
            return
        }
        process.terminationHandler = nil
        if process.isRunning {
            process.terminate()
        }
        self.process = nil
        status = .stopped
    }

    func restart() {
        stop()
        launch()
    }

    func scheduleLaunch(after seconds: TimeInterval) {
        cancelPendingLaunch()
        let start = max(1, Int(ceil(seconds)))
        pendingLaunchTask = Task { [weak self] in
            guard let self else { return }
            await MainActor.run {
                self.launchCountdownSeconds = start
            }
            var remaining = start
            while remaining > 0 {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                remaining -= 1
                if Task.isCancelled { return }
                await MainActor.run {
                    if remaining > 0 {
                        self.launchCountdownSeconds = remaining
                    } else {
                        self.launchCountdownSeconds = nil
                        self.launchIfNeeded()
                    }
                }
            }
        }
    }

    func cancelPendingLaunch() {
        pendingLaunchTask?.cancel()
        pendingLaunchTask = nil
        launchCountdownSeconds = nil
    }

    /// Check if an external exo instance is already running on the configured port
    func checkForExternalInstance() async -> Bool {
        let url = apiBaseURL.appendingPathComponent("node_id")
        var request = URLRequest(url: url)
        request.timeoutInterval = 2.0
        request.cachePolicy = .reloadIgnoringLocalCacheData

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               (200..<300).contains(httpResponse.statusCode) {
                return true
            }
        } catch {
            // API not responding - no external instance
        }
        return false
    }

    /// Enter monitoring mode for an externally-managed exo instance
    func enterMonitoringMode() {
        cancelPendingLaunch()
        stop()
        isMonitoringExternal = true
        status = .monitoring
    }

    /// Exit monitoring mode
    func exitMonitoringMode() {
        isMonitoringExternal = false
        if status == .monitoring {
            status = .stopped
        }
    }

    /// Smart launch that checks for external instance first
    func smartLaunch() async {
        if await checkForExternalInstance() {
            enterMonitoringMode()
        } else {
            launch()
        }
    }

    func revealRuntimeDirectory() {
        guard let runtimeDirectoryURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([runtimeDirectoryURL])
    }

    func statusTintColor() -> NSColor {
        switch status {
        case .running:
            return .systemGreen
        case .monitoring:
            return .systemBlue
        case .starting:
            return .systemYellow
        case .failed:
            return .systemRed
        case .stopped:
            return .systemGray
        }
    }

    private func resolveRuntimeDirectory() throws -> URL {
        let fileManager = FileManager.default

        if let override = ProcessInfo.processInfo.environment["EXO_RUNTIME_DIR"] {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            if fileManager.fileExists(atPath: url.path) {
                return url
            }
        }

        if let resourceRoot = Bundle.main.resourceURL {
            let bundled = resourceRoot.appendingPathComponent("exo", isDirectory: true)
            if fileManager.fileExists(atPath: bundled.path) {
                return bundled
            }
        }

        let repoCandidate = URL(fileURLWithPath: fileManager.currentDirectoryPath)
            .appendingPathComponent("dist/exo", isDirectory: true)
        if fileManager.fileExists(atPath: repoCandidate.path) {
            return repoCandidate
        }

        throw RuntimeError("Unable to locate the packaged EXO runtime.")
    }

    private func makeEnvironment(for runtimeURL: URL) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        environment["EXO_RUNTIME_DIR"] = runtimeURL.path
        environment["EXO_LIBP2P_NAMESPACE"] = computeNamespace()
        if !hfToken.isEmpty {
            environment["HF_TOKEN"] = hfToken
        }
        if enableImageModels {
            environment["EXO_ENABLE_IMAGE_MODELS"] = "true"
        }

        var paths: [String] = []
        if let existing = environment["PATH"], !existing.isEmpty {
            paths = existing.split(separator: ":").map(String.init)
        }

        let required = [
            runtimeURL.path,
            runtimeURL.appendingPathComponent("_internal").path,
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        ]

        for entry in required.reversed() {
            if !paths.contains(entry) {
                paths.insert(entry, at: 0)
            }
        }

        environment["PATH"] = paths.joined(separator: ":")
        return environment
    }

    private func buildTag() -> String {
        if let tag = Bundle.main.infoDictionary?["EXOBuildTag"] as? String, !tag.isEmpty {
            return tag
        }
        if let short = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
            !short.isEmpty
        {
            return short
        }
        return "dev"
    }

    private func computeNamespace() -> String {
        let base = buildTag()
        let custom = customNamespace.trimmingCharacters(in: .whitespaces)
        return custom.isEmpty ? base : custom
    }
}

struct RuntimeError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}
