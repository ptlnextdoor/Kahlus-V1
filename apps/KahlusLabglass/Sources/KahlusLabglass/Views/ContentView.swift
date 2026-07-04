import SwiftUI

struct ContentView: View {
    @ObservedObject var store: ArtifactStore

    var body: some View {
        NavigationSplitView {
            SidebarView(store: store)
        } detail: {
            DetailView(store: store)
        }
        .navigationSplitViewStyle(.balanced)
        .frame(minWidth: 1120, minHeight: 720)
        .background(LabColor.background)
        .toolbar {
            ToolbarItemGroup {
                Button {
                    store.openEvidencePackage()
                } label: {
                    Label("Open Evidence", systemImage: "folder.badge.gearshape")
                }
                Picker("View", selection: $store.page) {
                    ForEach(LabPage.allCases) { page in
                        Label(page.rawValue, systemImage: page.symbol).tag(page)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 520)
            }
        }
    }
}

struct SidebarView: View {
    @ObservedObject var store: ArtifactStore

    var body: some View {
        List(selection: $store.selectionID) {
            Section {
                Button {
                    store.openEvidencePackage()
                } label: {
                    Label("Open Evidence...", systemImage: "folder")
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(store.reports.count) indexed artifacts")
                        .font(.caption.weight(.semibold))
                    Text(store.index.root)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                if let message = store.loadMessage {
                    Text(message)
                        .font(.caption2)
                        .foregroundStyle(LabColor.muted)
                        .lineLimit(2)
                }
            }

            Section("Pages") {
                ForEach(LabPage.allCases) { page in
                    Button {
                        store.page = page
                    } label: {
                        Label(page.rawValue, systemImage: page.symbol)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(store.page == page ? LabColor.navy : .secondary)
                }
            }

            Section("Runs") {
                ForEach(store.reports) { report in
                    ArtifactSidebarRow(report: report)
                        .tag(report.id)
                }
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("Evidence")
        .navigationSplitViewColumnWidth(min: 250, ideal: 290)
    }
}

struct ArtifactSidebarRow: View {
    let report: ArtifactReport

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(report.statusColor)
                .frame(width: 9, height: 9)
            VStack(alignment: .leading, spacing: 2) {
                Text(report.title)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                Text(report.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 3)
    }
}

struct DetailView: View {
    @ObservedObject var store: ArtifactStore

    var body: some View {
        VStack(spacing: 0) {
            TopStrip(report: store.selected, index: store.index)
            Divider()
            ScrollView {
                VStack(spacing: 18) {
                    if let report = store.selected {
                        switch store.page {
                        case .overview:
                            OverviewPage(report: report)
                        case .signal:
                            SignalPage(report: report)
                        case .evidence:
                            EvidencePage(report: report)
                        case .mechanism:
                            MechanismPage(report: report)
                        case .ledger:
                            LedgerPage(index: store.index)
                        }
                    } else {
                        EmptyStateView(root: store.index.root)
                    }
                }
                .padding(22)
            }
            .background(LabColor.background)
        }
    }
}

struct TopStrip: View {
    let report: ArtifactReport?
    let index: ArtifactIndex

    var body: some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Kahlus Labglass")
                    .font(.system(.title2, design: .default).weight(.semibold))
                    .foregroundStyle(LabColor.navy)
                Text(report?.claimScope ?? "No artifact loaded")
                    .font(.subheadline)
                    .foregroundStyle(LabColor.muted)
                    .lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Text(index.root)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Text("Generated \(index.generatedAt)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if let report {
                StatusBadge(text: report.status, color: report.statusColor)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .background(LabColor.panel)
    }
}

struct OverviewPage: View {
    let report: ArtifactReport

    var body: some View {
        Grid(alignment: .topLeading, horizontalSpacing: 18, verticalSpacing: 18) {
            GridRow {
                StatusCard(report: report)
                    .gridCellColumns(1)
                PredicateCard(report: report)
                    .gridCellColumns(2)
            }
            GridRow {
                EvidenceTable(report: report)
                    .gridCellColumns(2)
                ControlsCard(report: report)
            }
            GridRow {
                InformationAtlasCard(report: report)
                    .gridCellColumns(3)
            }
        }
    }
}

struct SignalPage: View {
    let report: ArtifactReport

    var body: some View {
        Grid(horizontalSpacing: 18, verticalSpacing: 18) {
            GridRow {
                LabCard("Time-Frequency Thermal Map") {
                    ThermalMapView(seed: report.id.hashValue)
                        .frame(height: 300)
                }
                LabCard("Scalp Topomap") {
                    ScalpMapView(seed: report.id.hashValue)
                        .frame(height: 300)
                }
            }
            GridRow {
                LabCard("EEG Traces + Event Markers") {
                    TraceView(seed: report.id.hashValue)
                        .frame(height: 280)
                }
                .gridCellColumns(2)
            }
        }
    }
}

struct EvidencePage: View {
    let report: ArtifactReport

    var body: some View {
        VStack(spacing: 18) {
            EvidenceTable(report: report)
            HStack(alignment: .top, spacing: 18) {
                ControlsCard(report: report)
                ClaimBoundaryCard(report: report)
            }
        }
    }
}

struct MechanismPage: View {
    let report: ArtifactReport

    var body: some View {
        VStack(spacing: 18) {
            InformationAtlasCard(report: report)
            HStack(spacing: 18) {
                LabCard("Criticality / PIC Heat") {
                    ThermalMapView(seed: report.id.hashValue + 31)
                        .frame(height: 300)
                }
                LabCard("Channel Contribution") {
                    ScalpMapView(seed: report.id.hashValue + 17)
                        .frame(height: 300)
                }
            }
        }
    }
}

struct LedgerPage: View {
    let index: ArtifactIndex

    var body: some View {
        LabCard("Artifact Ledger") {
            VStack(alignment: .leading, spacing: 0) {
                LedgerHeader()
                ForEach(index.reports) { report in
                    LedgerRow(report: report)
                    Divider()
                }
            }
        }
    }
}

struct EmptyStateView: View {
    let root: String

    var body: some View {
        LabCard("No Artifacts") {
            Text("No gate, summary, or metric artifacts were indexed from \(root).")
                .foregroundStyle(LabColor.muted)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 60)
        }
    }
}
