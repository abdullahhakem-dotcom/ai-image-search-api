import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from transformers import pipeline

print("Loading Official AI Search Engine Pipeline...")
search_pipeline = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

GALLERY_PATH = "./my_gallery_2020_2021"

def run_search():
    query_text = search_entry.get().strip()
    if not query_text:
        return
        
    status_label.config(text=f"Searching for '{query_text}'...", foreground="blue")
    root.update()
    
    supported_formats = (".png", ".jpg", ".jpeg", ".webp")
    image_files = [f for f in os.listdir(GALLERY_PATH) if f.lower().endswith(supported_formats)]
    
    if not image_files:
        status_label.config(text="No images found in gallery!", foreground="red")
        return

    results = []
    for filename in image_files:
        file_path = os.path.join(GALLERY_PATH, filename)
        try:
            image = Image.open(file_path).convert("RGB")
            prediction = search_pipeline(image, candidate_labels=[query_text, "other items"])
            for pred in prediction:
                if pred['label'] == query_text:
                    results.append((filename, file_path, pred['score']))
                    break
        except:
            pass

    results.sort(key=lambda x: x[2], reverse=True)

    if results:
        top_filename, top_file_path, top_score = results[0]
        match_percentage = int(top_score * 100)
        
        status_label.config(text=f"Best Match: {top_filename} ({match_percentage}% Confidence)", foreground="green")
        
        # Display the image inside the App Window
        try:
            img = Image.open(top_file_path)
            img.thumbnail((400, 400)) # Scale it to fit the window nicely
            img_tk = ImageTk.PhotoImage(img)
            image_container.config(image=img_tk)
            image_container.image = img_tk
            
            # Setup button to reveal file in Windows Explorer
            reveal_button.config(command=lambda: os.system(f'explorer /select,"{os.path.abspath(top_file_path)}"'), state="normal")
        except Exception as e:
            status_label.config(text=f"Error displaying image: {e}", foreground="red")
    else:
        status_label.config(text="No matches found.", foreground="red")

# --- Setup App Window Graphics ---
root = tk.Tk()
root.title("AI Past Image Finder")
root.geometry("500x650")
root.attributes("-topmost", True) # Force window to stay on top

# Input Area
frame = ttk.Frame(root, padding="10")
frame.pack(fill="x")

search_entry = ttk.Entry(frame, font=("Arial", 12))
search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
search_entry.bind("<Return>", lambda e: run_search()) # Press Enter to search

search_button = ttk.Button(frame, text="Search", command=run_search)
search_button.pack(side="right")

# Status Label
status_label = ttk.Label(root, text="Type a description above and press Search", font=("Arial", 10, "italic"), padding="5")
status_label.pack()

# Image Display Panel
image_container = ttk.Label(root, text="[ Image Result Will Show Here ]", anchor="center", background="#f0f0f0")
image_container.pack(fill="both", expand=True, padx=20, pady=10)

# Reveal Button
reveal_button = ttk.Button(root, text="Open File Location", state="disabled")
reveal_button.pack(pady=15)

root.mainloop()
