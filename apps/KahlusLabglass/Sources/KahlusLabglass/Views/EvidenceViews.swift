import SwiftUI

struct EvidenceTable: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Model Evidence") {
            VStack(spacing: 0) {
                EvidenceHeader()
                ForEach(report.metrics.prefix(10)) { row in
                    EvidenceRow(row: row, maxValue: maxMetric)
                    Divider()
                }
                if report.metrics.isEmpty {
                    Text("No model metrics found in this artifact.")
                        .foregroundStyle(LabColor.muted)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 18)
                }
            }
        }
    }

    private var maxMetric: Double {
        max(0.0001, report.metrics.map { abs($0.value) }.max() ?? 0.0001)
    }
}

struct EvidenceHeader: View {
    var body: some View {
        HStack {
            Text("Model").frame(maxWidth: .infinity, alignment: .leading)
            Text("Value").frame(width: 92, alignment: .trailing)
            Text("CI").frame(width: 132, alignment: .leading)
            Text("Events").frame(width: 70, alignment: .trailing)
            Text("Baseline").frame(width: 160, alignment: .leading)
        }
        .font(.caption.weight(.bold))
        .foregroundStyle(LabColor.muted)
        .padding(.bottom, 8)
    }
}

struct EvidenceRow: View {
    let row: MetricRow
    let maxValue: Double

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(row.model)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(LabColor.navy)
                    .lineLimit(1)
                GeometryReader { proxy in
                    ZStack(alignment: .leading) {
                        Capsule().fill(LabColor.tint)
                        Capsule()
                            .fill(row.value >= 0 ? LabColor.blue : LabColor.failure)
                            .frame(width: max(3, proxy.size.width * min(1.0, abs(row.value) / maxValue)))
                    }
                }
                .frame(height: 8)
            }
            Text(fixed(row.value))
                .font(.system(.callout, design: .monospaced))
                .frame(width: 92, alignment: .trailing)
            Text(ciText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(LabColor.muted)
                .frame(width: 132, alignment: .leading)
            Text(row.events.map(String.init) ?? "-")
                .font(.caption)
                .frame(width: 70, alignment: .trailing)
            Text(shortText(row.baseline, to: 24))
                .font(.caption)
                .foregroundStyle(LabColor.muted)
                .frame(width: 160, alignment: .leading)
        }
        .padding(.vertical, 10)
    }

    private var ciText: String {
        if let ciLow = row.ciLow, let ciHigh = row.ciHigh {
            return "[\(fixed(ciLow, digits: 3)), \(fixed(ciHigh, digits: 3))]"
        }
        return "n/a"
    }
}

struct ControlsCard: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Controls") {
            VStack(spacing: 18) {
                ForEach(report.controls) { row in
                    HStack(spacing: 12) {
                        Text(row.name)
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(LabColor.navy)
                            .frame(width: 130, alignment: .leading)
                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Capsule().fill(LabColor.tint)
                                Capsule()
                                    .fill(row.kind == "true" ? LabColor.blue : LabColor.neutral)
                                    .frame(width: max(3, proxy.size.width * min(1.0, abs(row.value) / maxControl)))
                            }
                        }
                        .frame(height: 12)
                        Text(fixed(row.value))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(LabColor.muted)
                            .frame(width: 76, alignment: .trailing)
                    }
                }
            }
        }
    }

    private var maxControl: Double {
        max(0.0001, report.controls.map { abs($0.value) }.max() ?? 0.0001)
    }
}

struct InformationAtlasCard: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Conceptual Information Atlas") {
            VStack(alignment: .leading, spacing: 12) {
                GeometryReader { proxy in
                    HStack(spacing: 2) {
                        AtlasSegment(label: "I(Y;B)", color: LabColor.neutral, width: proxy.size.width * 0.28)
                        AtlasSegment(label: "I(Y;S|B)", color: LabColor.warning, width: proxy.size.width * 0.14)
                        AtlasSegment(label: "I(Y;X|B,S)", color: LabColor.blue, width: proxy.size.width * 0.42)
                        AtlasSegment(label: "Transfer", color: report.passed ? LabColor.success : LabColor.neutral, width: proxy.size.width * 0.16)
                    }
                }
                .frame(height: 42)
                Text("Conceptual guide only; segment widths are not measured decomposition values.")
                    .font(.callout)
                    .foregroundStyle(LabColor.muted)
            }
        }
    }
}

struct AtlasSegment: View {
    let label: String
    let color: Color
    let width: CGFloat

    var body: some View {
        Text(label)
            .font(.caption.weight(.bold))
            .foregroundStyle(color == LabColor.warning ? LabColor.navy : .white)
            .frame(width: width, height: 42)
            .background(color)
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct LedgerHeader: View {
    var body: some View {
        HStack {
            Text("Artifact").frame(maxWidth: .infinity, alignment: .leading)
            Text("Gate").frame(width: 90, alignment: .leading)
            Text("Scope").frame(width: 260, alignment: .leading)
        }
        .font(.caption.weight(.bold))
        .foregroundStyle(LabColor.muted)
        .padding(.bottom, 8)
    }
}

struct LedgerRow: View {
    let report: ArtifactReport

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(report.artifactName)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(LabColor.navy)
                    .lineLimit(1)
                Text(report.sourcePath)
                    .font(.caption)
                    .foregroundStyle(LabColor.muted)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            StatusBadge(text: report.status, color: report.statusColor)
                .frame(width: 90, alignment: .leading)
            Text(shortText(report.claimScope, to: 34))
                .font(.caption)
                .foregroundStyle(LabColor.muted)
                .frame(width: 260, alignment: .leading)
        }
        .padding(.vertical, 10)
    }
}
