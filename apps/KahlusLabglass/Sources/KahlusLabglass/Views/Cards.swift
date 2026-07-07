import SwiftUI

struct LabCard<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title)
                .font(.headline)
                .foregroundStyle(LabColor.navy)
            content
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(LabColor.panel)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(LabColor.border))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct StatusBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.headline.weight(.bold))
            .foregroundStyle(LabColor.navy)
            .padding(.horizontal, 16)
            .padding(.vertical, 9)
            .background(color.opacity(0.16))
            .overlay(Capsule().stroke(color, lineWidth: 1.2))
            .clipShape(Capsule())
    }
}

struct StatusCard: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Gate Status") {
            Text(report.status)
                .font(.system(size: 42, weight: .bold, design: .rounded))
                .foregroundStyle(report.statusColor)
            Text(report.failures.isEmpty ? "No gate failures recorded" : report.failures.joined(separator: ", "))
                .font(.callout)
                .foregroundStyle(LabColor.muted)
                .lineLimit(3)
            HStack(spacing: 8) {
                StatusBadge(text: report.metricUnit, color: LabColor.blue)
                StatusBadge(text: report.milestone, color: LabColor.neutral)
            }
        }
        .frame(minHeight: 190)
    }
}

struct PredicateCard: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Claim Gate Predicate") {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 3), spacing: 10) {
                PredicatePill(label: "Split", state: report.gatePredicate.split)
                PredicatePill(label: "Finite", state: report.gatePredicate.finite)
                PredicatePill(label: "Baseline", state: report.gatePredicate.baseline)
                PredicatePill(label: "Controls", state: report.gatePredicate.controls)
                PredicatePill(label: "Power", state: report.gatePredicate.power)
                PredicatePill(label: "Scope", state: report.gatePredicate.scope)
            }
            VStack(alignment: .leading, spacing: 7) {
                ForEach(report.metadata.prefix(6)) { row in
                    HStack {
                        Circle().fill(row.state.color).frame(width: 7, height: 7)
                        Text(row.label).foregroundStyle(LabColor.muted)
                        Spacer()
                        Text(shortText(row.value, to: 42)).foregroundStyle(LabColor.navy)
                    }
                    .font(.caption)
                }
            }
        }
        .frame(minHeight: 190)
    }
}

struct PredicatePill: View {
    let label: String
    let state: StatusKind

    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(state.color).frame(width: 8, height: 8)
            Text(label).font(.callout.weight(.semibold))
        }
        .foregroundStyle(LabColor.navy)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(state.color.opacity(0.12))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(state.color.opacity(0.65)))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct ClaimBoundaryCard: View {
    let report: ArtifactReport

    var body: some View {
        LabCard("Claim Boundary") {
            VStack(alignment: .leading, spacing: 10) {
                BoundaryLine(label: "Claim scope", value: report.claimScope)
                BoundaryLine(label: "Validation", value: report.validationScope)
                BoundaryLine(label: "Public data", value: report.publicDataUsed ? "yes" : "no")
                BoundaryLine(label: "External generalization", value: report.externalGeneralization ? "yes" : "no")
                BoundaryLine(label: "Nuisance conditioned", value: report.nuisanceConditioned ? "yes" : "no")
            }
        }
    }
}

struct BoundaryLine: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(LabColor.muted)
                .frame(width: 160, alignment: .leading)
            Text(value)
                .foregroundStyle(LabColor.navy)
                .fontWeight(.semibold)
            Spacer()
        }
        .font(.callout)
    }
}
