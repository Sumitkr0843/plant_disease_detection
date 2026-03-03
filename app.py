import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import predict

def upload_image():

    file_path = filedialog.askopenfilename()

    if file_path:

        img = Image.open(file_path)
        img = img.resize((300,300))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        disease = predict(file_path)

        result_label.config(text="Prediction: " + disease)

app = tk.Tk()
app.title("Plant Disease Detection")
app.geometry("500x500")

btn = tk.Button(app, text="Upload Leaf Image",
                command=upload_image)
btn.pack(pady=20)

panel = tk.Label(app)
panel.pack()

result_label = tk.Label(app, text="", font=("Arial",16))
result_label.pack(pady=20)

app.mainloop()