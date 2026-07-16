import { Tensor } from 'onnxruntime-web';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Prediction {
  box: BoundingBox;
  class_id: number;
  class_name: string;
  confidence: number;
}

const CLASS_NAMES = ["Fullripe", "Semiripe", "Unripe"];

/**
 * Preprocess the canvas image data into a Tensor for ONNX Runtime.
 * Assumes the canvas context provided is exactly modelWidth x modelHeight.
 */
export function preprocess(
  context: CanvasRenderingContext2D,
  modelWidth: number,
  modelHeight: number
): Tensor {
  const imgData = context.getImageData(0, 0, modelWidth, modelHeight);
  const data = imgData.data;

  // Float32Array to hold our normalized tensor data
  const float32Data = new Float32Array(3 * modelWidth * modelHeight);

  // YOLOv8 expects an input of shape [1, 3, 640, 640] 
  // with pixel values normalized to 0.0 - 1.0.
  // The layout is RRR... GGG... BBB...
  for (let i = 0; i < modelWidth * modelHeight; i++) {
    float32Data[i] = data[i * 4] / 255.0; // Red
    float32Data[modelWidth * modelHeight + i] = data[i * 4 + 1] / 255.0; // Green
    float32Data[2 * modelWidth * modelHeight + i] = data[i * 4 + 2] / 255.0; // Blue
  }

  return new Tensor('float32', float32Data, [1, 3, modelHeight, modelWidth]);
}

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes.
 */
function iou(box1: BoundingBox, box2: BoundingBox): number {
  const x1 = Math.max(box1.x1, box2.x1);
  const y1 = Math.max(box1.y1, box2.y1);
  const x2 = Math.min(box1.x2, box2.x2);
  const y2 = Math.min(box1.y2, box2.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

  return intersection / (area1 + area2 - intersection);
}

/**
 * Postprocess the output tensor from YOLOv8 to extract bounding boxes.
 * Includes Non-Maximum Suppression (NMS).
 */
export function postprocess(
  tensor: Tensor,
  modelWidth: number,
  modelHeight: number,
  confThreshold: number = 0.25,
  iouThreshold: number = 0.45
): Prediction[] {
  const output = tensor.data as Float32Array;
  const numClasses = CLASS_NAMES.length;
  const numElements = 4 + numClasses; // 4 bbox coordinates + 3 classes = 7
  const numAnchors = output.length / numElements; // Should be 8400 for 640x640
  
  // The output tensor from YOLOv8 ONNX export has shape [1, 7, 8400]
  // The memory layout is flattened:
  // 8400 x_centers, 8400 y_centers, 8400 widths, 8400 heights,
  // 8400 class_0_confs, 8400 class_1_confs, 8400 class_2_confs.
  
  let candidates: Prediction[] = [];

  for (let i = 0; i < numAnchors; i++) {
    // Determine the max class confidence for this anchor
    let maxConf = 0;
    let maxClassId = -1;
    for (let c = 0; c < numClasses; c++) {
      const conf = output[(4 + c) * numAnchors + i];
      if (conf > maxConf) {
        maxConf = conf;
        maxClassId = c;
      }
    }

    if (maxConf > confThreshold) {
      // Extract bounding box (x, y, w, h are at index 0, 1, 2, 3)
      const x = output[0 * numAnchors + i];
      const y = output[1 * numAnchors + i];
      const w = output[2 * numAnchors + i];
      const h = output[3 * numAnchors + i];

      // Convert center_x, center_y, width, height to x1, y1, x2, y2
      // relative to the 640x640 image size.
      const x1 = Math.max(0, x - w / 2);
      const y1 = Math.max(0, y - h / 2);
      const x2 = Math.min(modelWidth, x + w / 2);
      const y2 = Math.min(modelHeight, y + h / 2);

      candidates.push({
        box: { x1, y1, x2, y2 },
        class_id: maxClassId,
        class_name: CLASS_NAMES[maxClassId],
        confidence: maxConf
      });
    }
  }

  // Non-Maximum Suppression (NMS)
  // Sort candidates by confidence descending
  candidates.sort((a, b) => b.confidence - a.confidence);
  const selected: Prediction[] = [];

  for (const candidate of candidates) {
    let shouldSelect = true;
    for (const s of selected) {
      if (iou(candidate.box, s.box) > iouThreshold) {
        shouldSelect = false;
        break;
      }
    }
    if (shouldSelect) {
      selected.push(candidate);
    }
  }

  return selected;
}
