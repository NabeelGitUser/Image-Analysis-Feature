"""
Crop Image Analysis - CLI Interface
Usage:
  python image_cli.py --image rice.jpg
  python image_cli.py --image rice.jpg --mode disease
  python image_cli.py --image rice.jpg --mode query
"""

import argparse
from image_analysis import CropImageAnalyzer


def print_section(title: str, content: str):
    """Pretty print a section"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(content)


def run_full_analysis(analyzer: CropImageAnalyzer):
    """Run complete analysis and display results"""
    result = analyzer.analyze_image()

    if "error" in result:
        print(f"❌ {result['error']}")
        return

    print_section("🌾 CROP IDENTIFICATION",  result['crop_identification'])
    print_section("🦠 DISEASE DETECTION",    result['disease_detection'])
    print_section("📋 CROP DESCRIPTION",     result['crop_description'])
    print_section("💊 TREATMENT ADVICE",     result['treatment_advice'])


def run_query_mode(analyzer: CropImageAnalyzer):
    """Interactive Q&A loop about the image"""
    print("\n" + "=" * 60)
    print("  💬 Image Q&A Mode — ask anything about the crop!")
    print("  Type 'full' for full analysis | 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\n❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break

        if not question:
            continue

        if question.lower() == 'full':
            run_full_analysis(analyzer)
            continue

        print("\n🤔 Analyzing...")
        answer = analyzer.ask_about_image(question)

        print("\n" + "-" * 60)
        print(f"💡 Answer:\n{answer}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="🌾 Crop Image Analyzer — Detect crop type, disease & get treatment advice"
    )

    parser.add_argument(
        '--image',
        required=True,
        help='Path to crop image (jpg, jpeg, png, webp)'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'crop', 'disease', 'treatment', 'query'],
        default='full',
        help=(
            'full: complete analysis (default) | '
            'crop: identify crop only | '
            'disease: detect disease only | '
            'treatment: get treatment only | '
            'query: interactive Q&A'
        )
    )
    parser.add_argument(
        '--model',
        default='llama3.2-vision',
        help='Ollama vision model (default: llama3.2-vision)'
    )

    args = parser.parse_args()

    # Initialize analyzer
    print("\n🚀 Initializing Crop Image Analyzer...")
    analyzer = CropImageAnalyzer(model_name=args.model)

    # Load image
    if not analyzer.load_image(args.image):
        return

    # Run selected mode
    if args.mode == 'full':
        run_full_analysis(analyzer)

    elif args.mode == 'crop':
        print_section("🌾 CROP IDENTIFICATION", analyzer.identify_crop())

    elif args.mode == 'disease':
        print_section("🦠 DISEASE DETECTION", analyzer.detect_disease())

    elif args.mode == 'treatment':
        print_section("💊 TREATMENT ADVICE", analyzer.get_treatment())

    elif args.mode == 'query':
        # Show quick analysis first, then open Q&A
        print("\n⚡ Quick disease check first...")
        print_section("🦠 DISEASE DETECTION", analyzer.detect_disease())
        run_query_mode(analyzer)


if __name__ == "__main__":
    main()