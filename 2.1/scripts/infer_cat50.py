import argparse
from pathlib import Path

from catreid.infer.recognizer import CatReIDRecognizer


def parse_args():
    p = argparse.ArgumentParser("Cat ReID Inference")

    # ====== 默认生产参数（你现在这套） ======
    p.add_argument(
        "--ckpt",
        type=str,
        default="best_cat50_P8K2_lr1p5e4_margin0p22_9h_final.pt",
        help="path to trained checkpoint",
    )
    p.add_argument(
        "--db",
        type=str,
        default="checkpoints/vector_db_cat50_final.pt",
        help="path to vector database",
    )
    p.add_argument(
        "--img",
        type=str,
        default="infer_samples/test_002.jpg",
        help="image to recognize",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.25,   
        help="distance threshold for matching",
    )

    # 以后可以加：
    # p.add_argument("--gap", type=float, default=0.05)

    return p.parse_args()


def main():
    args = parse_args()

    recognizer = CatReIDRecognizer(
        ckpt_path=args.ckpt,
        vector_db_path=args.db,
        threshold=args.threshold,
    )

    report = recognizer.recognize_report(args.img, top_k=5)

    print("\n========== ReID Report ==========")
    print("Image:", report["img"])
    print(f"Gallery size: {report['gallery_size']}")
    print(f"Threshold: {report['threshold']:.4f} | top_k={report['top_k']}")
    print(f"Embedding: dim={report['embedding_dim']} | norm={report['embedding_norm']:.4f}")

    print("\nTop-K:")
    for item in report["topk"]:
        print(f"  #{item['rank']}  id={item['id']}  dist={item['distance']:.6f}")

    print("\nDecision:")
    print(f"  match={report['match']} | is_new={report['is_new']} | reason={report['reason']}")
    print(f"  pred_id={report['pred_id']} | pred_distance={report['pred_distance']:.6f}")
    print(f"  action={report['action']}")
    if report["is_new"]:
        print(f"  new_id_suggestion={report['new_id_suggestion']}")
    print("=================================\n")
    
    print(f"Timing: embed={report['embed_time_ms']:.2f}ms | search={report['search_time_ms']:.2f}ms | total={report['total_time_ms']:.2f}ms")


if __name__ == "__main__":
    main()