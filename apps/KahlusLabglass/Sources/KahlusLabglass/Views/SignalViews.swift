import SwiftUI

struct ThermalMapView: View {
    let seed: Int

    var body: some View {
        Canvas { context, size in
            let cols = 56
            let rows = 22
            let cellWidth = size.width / CGFloat(cols)
            let cellHeight = size.height / CGFloat(rows)
            let phase = Double(abs(seed % 97)) / 37.0
            for x in 0..<cols {
                for y in 0..<rows {
                    let value = 0.35
                        + 0.24 * sin(Double(x) * 0.22 + phase)
                        + 0.20 * cos(Double(y) * 0.58)
                        + 0.16 * sin(Double(x + y) * 0.13)
                    let rect = CGRect(
                        x: CGFloat(x) * cellWidth,
                        y: CGFloat(y) * cellHeight,
                        width: cellWidth + 0.5,
                        height: cellHeight + 0.5
                    )
                    context.fill(Path(rect), with: .color(thermalColor(value)))
                }
            }
            context.stroke(Path(CGRect(origin: .zero, size: size)), with: .color(LabColor.border), lineWidth: 1)
        }
        .overlay(alignment: .bottomLeading) {
            Text("time")
                .font(.caption)
                .foregroundStyle(LabColor.muted)
                .padding(8)
        }
        .overlay(alignment: .topLeading) {
            Text("frequency")
                .font(.caption)
                .foregroundStyle(LabColor.muted)
                .padding(8)
        }
    }
}

struct TraceView: View {
    let seed: Int

    var body: some View {
        Canvas { context, size in
            let channels = 5
            for line in 0...channels {
                let y = size.height * CGFloat(line) / CGFloat(channels)
                var grid = Path()
                grid.move(to: CGPoint(x: 0, y: y))
                grid.addLine(to: CGPoint(x: size.width, y: y))
                context.stroke(grid, with: .color(LabColor.border), lineWidth: 1)
            }
            for marker in [0.31, 0.66, 0.84] {
                let x = size.width * marker
                context.fill(Path(CGRect(x: x - 4, y: 0, width: 8, height: size.height)), with: .color(LabColor.failure.opacity(0.10)))
                var event = Path()
                event.move(to: CGPoint(x: x, y: 0))
                event.addLine(to: CGPoint(x: x, y: size.height))
                context.stroke(event, with: .color(LabColor.failure.opacity(0.68)), style: StrokeStyle(lineWidth: 1.2, dash: [4, 5]))
            }
            for channel in 0..<4 {
                var trace = Path()
                let base = 28 + CGFloat(channel) * (size.height - 44) / 4.0
                for point in 0..<190 {
                    let x = CGFloat(point) * size.width / 189.0
                    let y = base - CGFloat(sample(channel: channel, point: point)) * 24
                    if point == 0 {
                        trace.move(to: CGPoint(x: x, y: y))
                    } else {
                        trace.addLine(to: CGPoint(x: x, y: y))
                    }
                }
                context.stroke(trace, with: .color(LabColor.navy), lineWidth: 1.5)
                context.draw(Text("CH\(channel + 1)").font(.caption).foregroundColor(LabColor.muted), at: CGPoint(x: 20, y: base - 16))
            }
        }
    }

    private func sample(channel: Int, point: Int) -> Double {
        let phase = Double(abs(seed % 53)) / 19.0
        let p = Double(point)
        return sin(p * 0.10 + Double(channel) * 0.7 + phase) * (0.55 + Double(channel) * 0.04)
            + sin(p * 0.031 + Double(channel)) * 0.28
            + ([58, 118, 150].contains(point) ? 0.18 : 0.0)
    }
}

struct ScalpMapView: View {
    let seed: Int

    var body: some View {
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            let radius = side * 0.34
            let center = CGPoint(x: proxy.size.width / 2, y: proxy.size.height / 2 + 8)
            ZStack {
                Circle()
                    .fill(LabColor.background)
                    .overlay(Circle().stroke(LabColor.border, lineWidth: 2))
                    .frame(width: radius * 2, height: radius * 2)
                    .position(center)
                Capsule()
                    .stroke(LabColor.border, lineWidth: 2)
                    .frame(width: 34, height: 16)
                    .position(x: center.x, y: center.y - radius - 4)
                ForEach(0..<8, id: \.self) { idx in
                    let angle = Double(idx) * 45.0 + 18.0
                    let point = CGPoint(
                        x: center.x + cos(angle * .pi / 180) * radius * 0.58,
                        y: center.y + sin(angle * .pi / 180) * radius * 0.58
                    )
                    let value = abs(sin(Double(idx + abs(seed % 11)) * 0.73))
                    Circle()
                        .fill(mix(Color(red: 0.145, green: 0.388, blue: 0.922), LabColor.failure, value))
                        .overlay(Circle().stroke(.white, lineWidth: 2))
                        .frame(width: 28, height: 28)
                        .position(point)
                    Text("C\(idx + 1)")
                        .font(.caption2)
                        .foregroundStyle(LabColor.muted)
                        .position(x: point.x, y: point.y + 24)
                }
            }
        }
    }
}
