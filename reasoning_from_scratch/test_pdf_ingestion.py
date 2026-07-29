import sys
sys.path.append(r"C:\Users\Immanuel\Desktop\reasoning-from-scratch\chG\01_main-chapter-code")

from qwen3_chat_interface_multiturn import load_pdf

if __name__ == "__main__":
    file_path = r"C:\Users\Immanuel\Documents\CamScanner 11-07-2025 13.41.pdf"
    texts = load_pdf(file_path)

    print("\n--- Extracted Text Preview ---")
    for i, t in enumerate(texts[:2]):   # show first 2 pages only
        print(f"Page {i+1}: {t[:300]}...\n")   # print first 300 characters
