import Foundation
import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ArtifactIndex: Decodable {
    let generatedAt: String
    let root: String
    let reports: [ArtifactReport]
}

struct ArtifactReport: Decodable, Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let artifactName: String
    let sourcePath: String
    let milestone: String
    let status: String
    let statusKind: StatusKind
    let passed: Bool
    let failures: [String]
    let claimScope: String
    let validationScope: String
    let publicDataUsed: Bool
    let externalGeneralization: Bool
    let nuisanceConditioned: Bool
    let metricUnit: String
    let metrics: [MetricRow]
    let controls: [ControlRow]
    let gatePredicate: GatePredicate
    let metadata: [MetaRow]

    var statusColor: Color {
        statusKind.color
    }
}

enum StatusKind: String, Decodable {
    case pass
    case fail
    case warning
    case neutral

    var color: Color {
        switch self {
        case .pass: LabColor.success
        case .fail: LabColor.failure
        case .warning: LabColor.warning
        case .neutral: LabColor.neutral
        }
    }
}

struct MetricRow: Decodable, Hashable, Identifiable {
    var id: String { model }
    let model: String
    let value: Double
    let ciLow: Double?
    let ciHigh: Double?
    let events: Int?
    let baseline: String
    let kind: String
}

struct ControlRow: Decodable, Hashable, Identifiable {
    var id: String { name }
    let name: String
    let value: Double
    let kind: String
}

struct MetaRow: Decodable, Hashable, Identifiable {
    var id: String { label }
    let label: String
    let value: String
    let state: StatusKind
}

struct GatePredicate: Decodable, Hashable {
    let split: StatusKind
    let finite: StatusKind
    let baseline: StatusKind
    let controls: StatusKind
    let power: StatusKind
    let scope: StatusKind
}

@MainActor
final class ArtifactStore: ObservableObject {
    @Published var index: ArtifactIndex
    @Published var selectionID: ArtifactReport.ID?
    @Published var page: LabPage = .overview
    @Published var loadMessage: String?

    var reports: [ArtifactReport] {
        index.reports
    }

    var selected: ArtifactReport? {
        if let selectionID, let report = reports.first(where: { $0.id == selectionID }) {
            return report
        }
        return reports.first
    }

    init(index: ArtifactIndex) {
        self.index = index
        self.selectionID = index.reports.first?.id
    }

    static func load() -> ArtifactStore {
        let decoder = JSONDecoder()
        if let url = Bundle.main.url(forResource: "artifact-index", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let index = try? decoder.decode(ArtifactIndex.self, from: data) {
            return ArtifactStore(index: index)
        }
        return ArtifactStore(index: ArtifactIndex(generatedAt: "not generated", root: "none", reports: []))
    }

    func openEvidencePackage() {
        let panel = NSOpenPanel()
        panel.title = "Open Kahlus Evidence Folder or Package"
        panel.message = "Choose a versions folder, evidence bundle .zip, or gate JSON artifact."
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.zip, .json]
        if panel.runModal() == .OK, let url = panel.url {
            load(from: url)
        }
    }

    func load(from url: URL) {
        guard let script = Self.indexerScriptURL() else {
            loadMessage = "Missing bundled indexer."
            return
        }
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("kahlus-labglass-\(UUID().uuidString).json")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [script.path, url.path, outputURL.path]
        let pipe = Pipe()
        process.standardError = pipe
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                let error = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "indexer failed"
                loadMessage = error
                return
            }
            let data = try Data(contentsOf: outputURL)
            let nextIndex = try JSONDecoder().decode(ArtifactIndex.self, from: data)
            index = nextIndex
            selectionID = nextIndex.reports.first?.id
            loadMessage = "Loaded \(nextIndex.reports.count) artifacts from \(url.lastPathComponent)."
        } catch {
            loadMessage = error.localizedDescription
        }
    }

    private static func indexerScriptURL() -> URL? {
        if let bundled = Bundle.main.url(forResource: "build_labglass_index", withExtension: "py") {
            return bundled
        }
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<6 {
            let candidate = dir.appendingPathComponent("scripts/build_labglass_index.py")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            dir.deleteLastPathComponent()
        }
        return nil
    }
}

enum LabPage: String, CaseIterable, Identifiable {
    case overview = "Overview"
    case signal = "Signal"
    case evidence = "Evidence"
    case mechanism = "Mechanism"
    case ledger = "Ledger"

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .overview: "gauge.with.dots.needle.67percent"
        case .signal: "waveform.path.ecg"
        case .evidence: "chart.bar.xaxis"
        case .mechanism: "point.3.connected.trianglepath.dotted"
        case .ledger: "list.bullet.rectangle.portrait"
        }
    }
}

enum LabColor {
    static let background = Color.white
    static let panel = Color(red: 0.973, green: 0.984, blue: 1.0)
    static let tint = Color(red: 0.918, green: 0.965, blue: 1.0)
    static let border = Color(red: 0.847, green: 0.906, blue: 0.953)
    static let blue = Color(red: 0.220, green: 0.741, blue: 0.973)
    static let navy = Color(red: 0.059, green: 0.208, blue: 0.341)
    static let muted = Color(red: 0.392, green: 0.455, blue: 0.545)
    static let success = Color(red: 0.086, green: 0.639, blue: 0.290)
    static let warning = Color(red: 0.961, green: 0.620, blue: 0.043)
    static let failure = Color(red: 0.863, green: 0.149, blue: 0.149)
    static let neutral = Color(red: 0.580, green: 0.639, blue: 0.722)
    static let pale = Color(red: 0.937, green: 0.965, blue: 1.0)
    static let yellow = Color(red: 0.992, green: 0.902, blue: 0.541)
    static let orange = Color(red: 0.976, green: 0.451, blue: 0.086)
}
