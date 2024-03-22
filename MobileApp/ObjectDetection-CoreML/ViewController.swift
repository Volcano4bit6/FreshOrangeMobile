//
//  ViewController.swift
//  SSDMobileNet-CoreML
//
//  Created by GwakDoyoung on 01/02/2019.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit
import Vision
import CoreMedia

class ViewController: UIViewController {

    // MARK: - UI Properties
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var boxesView: DrawingBoundingBoxView!
    @IBOutlet weak var modelSelectionSegmentedControl: UISegmentedControl!
    
    @IBOutlet weak var inferenceLabel: UILabel!
    @IBOutlet weak var etimeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    @IBOutlet weak var captureButton: UIButton!
    // MARK - Core ML model
    // YOLOv3(iOS12+), YOLOv3FP16(iOS12+), YOLOv3Int8LUT(iOS12+)
    // YOLOv3Tiny(iOS12+), YOLOv3TinyFP16(iOS12+), YOLOv3TinyInt8LUT(iOS12+)
    // MobileNetV2_SSDLite(iOS12+), ObjectDetector(iOS12+)
    // yolov5n(iOS13+), yolov5s(iOS13+), yolov5m(iOS13+), yolov5l(iOS13+), yolov5x(iOS13+)
    // yolov5n6(iOS13+), yolov5s6(iOS13+), yolov5m6(iOS13+), yolov5l6(iOS13+), yolov5x6(iOS13+)
    // yolov8n(iOS14+), yolov8s(iOS14+), yolov8m(iOS14+), yolov8l(iOS14+), yolov8x(iOS14+)
//    lazy var objectDectectionModel = { return try? yolov8s() }()
    lazy var objectDetectionModels: [ MLModel? ] = [
        { return try? orangenanodet().model }(),
        { return try? orangeyolo().model }()
    ]

    
    // MARK: - Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var isInferencing = false
    
    // MARK: - AV Property
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 1)
    var lastExecution = Date()
    
    // MARK: - TableView Data
    var predictions: [VNRecognizedObjectObservation] = []
    
    // MARK - Performance Measurement Property
    private let 👨‍🔧 = 📏()
    
    let maf1 = MovingAverageFilter()
    let maf2 = MovingAverageFilter()
    let maf3 = MovingAverageFilter()

    var currentPixelBuffer: CVPixelBuffer? 
    
    @IBAction func modelSelectionChanged(_ sender: UISegmentedControl) {
        setUpModel(selectIndex: sender.selectedSegmentIndex)
    }
    
    @IBAction func onCaptureTap(_ sender: UIButton) {
        DispatchQueue.global(qos: .utility).async {
            guard let pixelBuffer = self.currentPixelBuffer else { return }
    //        print(pixelBuffer)
    //        let ciImage = CIImage(cvImageBuffer: currentPixelBuffer!)
            let rawImage = UIImage(pixelBuffer: pixelBuffer)
            // Save raw image
            self.saveImageToPhotoAlbum(rawImage!)
            if let imageWithBoundingBoxes = self.drawImageWithBoundingBoxes(rawImage!) {
                self.saveImageToPhotoAlbum(imageWithBoundingBoxes)
            }
        }
//        guard let pixelBuffer = self.currentPixelBuffer else { return }
////        print(pixelBuffer)
////        let ciImage = CIImage(cvImageBuffer: currentPixelBuffer!)
//        let rawImage = UIImage(pixelBuffer: pixelBuffer)
//        // Save raw image
//        self.saveImageToPhotoAlbum(rawImage!)
//        if let imageWithBoundingBoxes = self.drawImageWithBoundingBoxes(rawImage!) {
//            self.saveImageToPhotoAlbum(imageWithBoundingBoxes)
//        }
    }

    func drawImageWithBoundingBoxes(_ image: UIImage) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(image.size, true, 0)
        image.draw(at: .zero)
        
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        
        for prediction in self.predictions {
            let scale = CGAffineTransform.identity.scaledBy(x: image.size.width, y: image.size.height)
            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
            let rect = prediction.boundingBox.applying(transform).applying(scale)

            // Draw the bounding box
            let color = self.boxesView.labelColor(with: prediction.label ?? "N/A")
            context.setStrokeColor(color.cgColor)
            context.setLineWidth(4)
            context.stroke(rect)
            
            // Draw the label
            let labelString = prediction.label ?? "N/A"
            let confidence = prediction.confidence
            let confidenceString = String(format: "%.2f%%", confidence * 100)
            let combinedString = "\(labelString) \(confidenceString)"
            
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 13),
                .foregroundColor: UIColor.black,
                .backgroundColor: color
            ]
            
            let stringSize = combinedString.size(withAttributes: attributes)
            let labelRect = CGRect(x: rect.origin.x, y: rect.origin.y - stringSize.height, width: stringSize.width, height: stringSize.height)
            
            combinedString.draw(in: labelRect, withAttributes: attributes)
        }
        
        let imageWithBoundingBoxes = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return imageWithBoundingBoxes
    }
    // MARK: - View Controller Life Cycle
    override func viewDidLoad() {

        super.viewDidLoad()
        
//        modelSelectionSegmentedControl.insertSegment(withTitle: "Nanodet", at: 0, animated: false)
//        modelSelectionSegmentedControl.insertSegment(withTitle: "Yolov8", at: 1, animated: false)

        
//        if modelSelectionSegmentedControl.numberOfSegments > 1 {
//            modelSelectionSegmentedControl.setTitle("Nanodet", forSegmentAt: 0)
//            modelSelectionSegmentedControl.setTitle("Yolov8", forSegmentAt: 1)
//        } else {
//            // Handle the case where there aren't enough segments.
//            // You might want to add them programmatically if needed:
//        }
        
        // setup the model
        setUpModel()
        
        // setup camera
        setUpCamera()
        
        // setup delegate for performance measurement
        👨‍🔧.delegate = self
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        setUpCamera()
//        justSetupCamera()
//        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    // MARK: - Setup Core ML
    func setUpModel(selectIndex: Int = 0) {
//        let selectIndex = modelSelectionSegmentedControl.selectedSegmentIndex
        let model = objectDetectionModels[selectIndex]
        guard let objectDectectionModel = model else { fatalError("fail to load the model") }
        if let visionModel = try? VNCoreMLModel(for: objectDectectionModel) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("fail to create vision model")
        }
    }

    func justSetupCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        let _ = videoCapture.setUp(sessionPreset: .vga640x480)
    }

    // MARK: - SetUp Video
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
//        videoCapture.setUp(sessionPreset: .vga640x480) { success in
//
//            if success {
//                // add preview view on the layer
//                if let previewLayer = self.videoCapture.previewLayer {
//                    self.videoPreview.layer.addSublayer(previewLayer)
//                    self.resizePreviewLayer()
//                }
//
//                // start video preview when setup is done
//                self.videoCapture.start()
//            }
//        }
        let isSuccess = videoCapture.setUp(sessionPreset: .vga640x480)
        if isSuccess {
            if let previewLayer = videoCapture.previewLayer {
                self.videoPreview.layer.addSublayer(previewLayer)
                self.resizePreviewLayer()
            }
            self.videoCapture.start()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        resizePreviewLayer()
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
}

// MARK: - Save Image to Photo Album
extension ViewController {
    func saveImageToPhotoAlbum(_ image: UIImage) {
        // Ensure the image is not nil and save it to the photo album
        UIImageWriteToSavedPhotosAlbum(image, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        // This method is called once the image has been saved to the photo album
        if let error = error {
            // If an error occurred, print it out
            print("Error saving image to photo album: \(error)")
        } else {
            // If no error occurred, print a success message
            print("Image successfully saved to photo album")
        }
    }
}

// MARK: - VideoCaptureDelegate
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // the captured image from camera is contained on pixelBuffer
        self.currentPixelBuffer = pixelBuffer
        if !self.isInferencing, let pixelBuffer = pixelBuffer {
            self.isInferencing = true
            
            // start of measure
            self.👨‍🔧.🎬👏()
            
            // predict!
            self.predictUsingVision(pixelBuffer: pixelBuffer)
        }
    }
}

extension ViewController {
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        self.semaphore.wait()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
    
    // MARK: - Post-processing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        self.👨‍🔧.🏷(with: "endInference")
        if let predictions = request.results as? [VNRecognizedObjectObservation] {
//            print(predictions.first?.labels.first?.identifier ?? "nil")
//            print(predictions.first?.labels.first?.confidence ?? -1)
//            print(predictions)
            self.predictions = predictions
            DispatchQueue.main.async {
                self.boxesView.predictedObjects = predictions

                // end of measure
                self.👨‍🔧.🎬🤚()
                
                self.isInferencing = false
            }
        } else {
            // end of measure
            self.👨‍🔧.🎬🤚()
            
            self.isInferencing = false
        }
        self.semaphore.signal()
    }
}

extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return predictions.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = tableView.dequeueReusableCell(withIdentifier: "InfoCell") else {
            return UITableViewCell()
        }

        let rectString = predictions[indexPath.row].boundingBox.toString(digit: 2)
        let confidence = predictions[indexPath.row].labels.first?.confidence ?? -1
        let confidenceString = String(format: "%.3f", confidence/*Math.sigmoid(confidence)*/)
        
        cell.textLabel?.text = predictions[indexPath.row].label ?? "N/A"
        cell.detailTextLabel?.text = "\(rectString), \(confidenceString)"
        return cell
    }
}

// MARK: - 📏(Performance Measurement) Delegate
extension ViewController: 📏Delegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        //print(executionTime, fps)
        DispatchQueue.main.async {
            self.maf1.append(element: Int(inferenceTime*1000.0))
            self.maf2.append(element: Int(executionTime*1000.0))
            self.maf3.append(element: fps)
            
            self.inferenceLabel.text = "inference: \(self.maf1.averageValue) ms"
            self.etimeLabel.text = "execution: \(self.maf2.averageValue) ms"
            self.fpsLabel.text = "fps: \(self.maf3.averageValue)"
        }
    }
}

extension UIImage {
    convenience init?(pixelBuffer: CVPixelBuffer) {
        var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            self.init(cgImage: cgImage)
        } else {
            return nil
        }
    }
}

class MovingAverageFilter {
    private var arr: [Int] = []
    private let maxCount = 10
    
    public func append(element: Int) {
        arr.append(element)
        if arr.count > maxCount {
            arr.removeFirst()
        }
    }
    
    public var averageValue: Int {
        guard !arr.isEmpty else { return 0 }
        let sum = arr.reduce(0) { $0 + $1 }
        return Int(Double(sum) / Double(arr.count))
    }
}


