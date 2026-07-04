// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "KahlusLabglass",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "KahlusLabglass", targets: ["KahlusLabglass"])
    ],
    targets: [
        .executableTarget(name: "KahlusLabglass")
    ]
)
