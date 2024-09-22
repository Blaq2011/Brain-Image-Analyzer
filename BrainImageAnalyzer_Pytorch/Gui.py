import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # For image handling
from Test import predict_plane  # Import the function from test.py

# Function to handle file selection and display the image
def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
    
    if file_path:
        # Display the file path
        label_file_explorer.config(text="Filename: " + file_path)

        # Open and display the selected image
        img = Image.open(file_path)

        # Maintain aspect ratio while resizing
        img.thumbnail((200, 200))  # Resize with aspect ratio
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection

        # Call the predict_plane function and get the results
        predicted_class, confidence = predict_plane(file_path)

        # Display the prediction results on the interface
        result_label.config(text=f"Results: {predicted_class}\nConfidence: {confidence:.4f}")

# Initialize the main window
root = tk.Tk()
root.title("Image Classification")
root.geometry("600x350")  # Adjusted window size

# Create a button to select an image
button_explore = tk.Button(root, text="Select Image", command=select_image, font=("Helvetica", 12), bg="#4CAF50", fg="white")
button_explore.pack(pady=10)

# Label to display the selected file path (Filename)
label_file_explorer = tk.Label(root, text="No File Selected", width=80, fg="blue", font=("Helvetica", 10))
label_file_explorer.pack(pady=5)

# Label to display the prediction results
result_label = tk.Label(root, text="Prediction will appear here.", width=60, height=4, fg="black", font=("Helvetica", 12))
result_label.pack(pady=10)

# Label to display the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Start the tkinter loop
root.mainloop()
