from test.evaluate import evaluate
from test.evaluate import predict
import cv2
import os

if __name__ == "__main__":
    T2_ADR = r"https://drive.google.com/file/d/1mbUYsUqv6rP6ONC3rRDhly7RNBLZNp3Z/view"
    T4_ADR = r"https://drive.google.com/file/d/1SRE1iBAuLs1vjd5Bpb7WogFu6mcp6m6N/view"
    T5_ADR = None

    model_paths = {
        2: T2_ADR,
        4: T4_ADR,
        5: T5_ADR
    }

    image_path = r"C:\Users\Zachary Bamberger\Documents\Technion\Deep Learning\Final Project\example"

    im = evaluate(image_path, model_paths)
    print(im)
