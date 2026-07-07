import Foundation
import SwiftUI

func fixed(_ value: Double, digits: Int = 4) -> String {
    value.formatted(.number.precision(.fractionLength(digits)))
}

func shortText(_ value: String, to length: Int) -> String {
    value.count <= length ? value : String(value.prefix(max(0, length - 3))) + "..."
}

func thermalColor(_ value: Double) -> Color {
    let v = min(1.0, max(0.0, value))
    if v < 0.55 {
        return mix(LabColor.pale, LabColor.blue, v / 0.55)
    }
    if v < 0.82 {
        return mix(LabColor.blue, LabColor.yellow, (v - 0.55) / 0.27)
    }
    return mix(LabColor.yellow, LabColor.orange, (v - 0.82) / 0.18)
}

func mix(_ a: Color, _ b: Color, _ t: Double) -> Color {
    #if os(macOS)
    let ca = NSColor(a).usingColorSpace(.sRGB) ?? .white
    let cb = NSColor(b).usingColorSpace(.sRGB) ?? .white
    return Color(
        red: ca.redComponent + (cb.redComponent - ca.redComponent) * t,
        green: ca.greenComponent + (cb.greenComponent - ca.greenComponent) * t,
        blue: ca.blueComponent + (cb.blueComponent - ca.blueComponent) * t
    )
    #else
    return a
    #endif
}
