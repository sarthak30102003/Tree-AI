import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import webbrowser
import os
import cv2
from tkinter import Canvas, Scrollbar, Label, Frame, Entry
import numpy as np
from tensorflow.keras.models import load_model

class win:
    def __init__(self,win):
        
        win.geometry("810x500+100+100")
        win.resizable(False, False)
        win.title("Tree-AI Console")
        
        load = Image.open(r"gi\Main.jpg")
        self.bgimage = ImageTk.PhotoImage(load)
        load = Image.open(r"gi\Main2.jpg")
        self.bgimage2 = ImageTk.PhotoImage(load)
        self.back_label = ttk.Label(win,image=self.bgimage)
        self.back_label.place(x=-2,y=-2)
#################################################################################
########################       MODEL          ######################################
         
         # Load the trained model
        model = load_model('plant_disease_model.h5')  # Update with your model file path

        # Define the class mapping dictionary
        class_mapping = {
            0: ('Generic', 'Healthy'),
            1: ('Generic', 'Powdery'),
            2: ('Generic', 'Rusty'),
            3: ('CORN', 'Rust'),
            4: ('CORN', 'Gray Spot'),
            5: ('CORN', 'Healthy'),
            6: ('CORN', 'Leaf Blight'),
            7: ('Potato', 'Early Blight'),
            8: ('Potato', 'Healthy'),
            9: ('Potato', 'Late Blight'),
            10: ('SugarCane', 'Bacterial Blight'),
            11: ('SugarCane', 'Healthy'),
            12: ('SugarCane', 'Red Rot'),
            13: ('Wheat', 'Brown Rust'),
            14: ('Wheat', 'Healthy'),
            15: ('Wheat', 'Yellow Rust'),
        }

        # Specify the folder containing images for classification
        input_folder = "Test"
        
        # Create a canvas with a scrollbar
        canvas = Canvas(win, width=570, height=500, bg="#242424")
        scroll_y = Scrollbar(win, orient="vertical", command=canvas.yview, bg="#242424")
        scroll_y.pack(side="right", fill="y")
        # canvas.place(x=230,y=0)
        canvas.configure(yscrollcommand=scroll_y.set)
        
        # Create a frame inside the canvas to hold the images and labels
        frame = Frame(canvas, bg="#242424")
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Get a list of all image files in the folder
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Function to update the canvas with images and labels
        def update_images():
            for i, image_file in enumerate(image_files):
                # Read and preprocess the image
                image_path = os.path.join(input_folder, image_file)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (256, 256))  # Ensure the image size matches the model input size
                img_display = img.copy()
                img = img / 255.0  # Normalize pixel values

                # Expand dimensions to match the model input shape
                img = np.expand_dims(img, axis=0)

                # Make prediction
                prediction = model.predict(img)
                predicted_class = np.argmax(prediction)

                # Use the class_mapping dictionary to get species and disease names
                species, disease = class_mapping[predicted_class]

                # Display the image with predicted class information
                img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)

                # Create a frame for each image and label
                frame_image_label = Frame(frame, bd=2, relief="groove", bg="#242424")
                frame_image_label.grid(row=i, column=0, sticky="w", pady=5)

                label_image = Label(frame_image_label, image=img_tk, bg="#242424")
                label_image.grid(row=0, column=0, sticky="w", padx=10, pady=5)

                # Create a frame for output text with a white-to-gray gradient
                frame_output = Frame(frame_image_label, bd=2, relief="groove", bg="#242424")
                frame_output.grid(row=0, column=1, sticky="w", padx=10, pady=5)

                label_heading = Label(frame_output, text=f"Species: {species}\nDisease: {disease}", bg="#242424", fg="white")
                label_heading.grid(row=0, column=0, sticky="w", padx=5, pady=5)

                label_image.img_tk = img_tk  # Save reference to prevent the image from being garbage collected

        # Update the canvas with images and labels
        update_images()

        # Configure canvas scrolling
        frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        
################### MODEL END ################################################333
        def result(event=None):
            self.reportbutton.configure(image=self.reportbuttonimage)
            self.Homebutton.configure(image=self.homebuttonimage)
            self.healthbutton.configure(image=self.healthbuttonimage)
            self.resultbutton.configure(image=self.resultbuttonlightimage)
            self.back_label.configure(image=self.bgimage2)
            canvas.place(x=230,y=0)
            
            

        def home(event=None):
            self.reportbutton.configure(image=self.reportbuttonimage)
            self.Homebutton.configure(image=self.homebuttonlightimage)
            self.healthbutton.configure(image=self.healthbuttonimage)
            self.resultbutton.configure(image=self.resultbuttonimage)
            self.back_label.configure(image=self.bgimage)
            canvas.place_forget()
        def report(event=None):
            self.reportbutton.configure(image=self.reportbuttonlightimage)
            self.Homebutton.configure(image=self.homebuttonimage)
            self.healthbutton.configure(image=self.healthbuttonimage)
            self.resultbutton.configure(image=self.resultbuttonimage)
            self.back_label.configure(image=self.bgimage2)
            canvas.place_forget()
        
        def healthmap(event=None):
            self.reportbutton.configure(image=self.reportbuttonimage)
            self.Homebutton.configure(image=self.homebuttonimage)
            self.healthbutton.configure(image=self.healthbuttonlightimage)
            self.resultbutton.configure(image=self.resultbuttonimage)
            self.back_label.configure(image=self.bgimage2)
            canvas.place_forget()

        
        load = Image.open(r"gi\homebut.jpg")
        self.homebuttonimage = ImageTk.PhotoImage(load)

        load = Image.open(r"gi\homebutlight.jpg")
        self.homebuttonlightimage = ImageTk.PhotoImage(load)
        self.Homebutton = tk.Button(win,image = self.homebuttonlightimage,command = home,border=0,highlightthickness=0)
        self.Homebutton.place(x = 0,y=70)

        load = Image.open(r"gi\reportbut.jpg")
        self.reportbuttonimage = ImageTk.PhotoImage(load)
        self.reportbutton = tk.Button(win,image = self.reportbuttonimage,command = report,border=0,highlightthickness=0)
        self.reportbutton.place(x = 0,y=120)

        load = Image.open(r"gi\reportbutlight.jpg")
        self.reportbuttonlightimage = ImageTk.PhotoImage(load)


        load = Image.open(r"gi\resultbut.jpg")
        self.resultbuttonimage = ImageTk.PhotoImage(load)
        self.resultbutton = tk.Button(win,image = self.resultbuttonimage,command = result,border=0,highlightthickness=0)
        self.resultbutton.place(x = 0,y=170)

        load = Image.open(r"gi\resultbutlight.jpg")
        self.resultbuttonlightimage = ImageTk.PhotoImage(load)


        load = Image.open(r"gi\healthbut.jpg")
        self.healthbuttonimage = ImageTk.PhotoImage(load)
        self.healthbutton = tk.Button(win,image = self.healthbuttonimage,command = healthmap,border=0,highlightthickness=0)
        self.healthbutton.place(x = 0,y=220)

        load = Image.open(r"gi\healthbutlight.jpg")
        self.healthbuttonlightimage = ImageTk.PhotoImage(load)




        crevar = tk.StringVar()
        def credit(self):
            cre = crevar.get()
            if cre == "Roshan":
                open_link("https://www.linkedin.com/in/roshan-kumar-dubey-620253260")
            elif cre == "Rupali":
                open_link("https://www.linkedin.com/in/rupali-mandawat-93b265255/")
            elif cre == "Sarthak":
                open_link("https://www.linkedin.com/in/sarthak-aggarwal-486b60240/")
            elif cre == "Sumit":
                open_link("https://www.linkedin.com/in/sumit-kumar-sengar-3a2245147")
            elif cre=="Harshit":
                open_link("https://i.pinimg.com/736x/fd/01/75/fd0175ef780e2feefb30055be9f2e022.jpg")
        def open_link(link):
            webbrowser.open(link)

        self.credits= ttk.Combobox(win, textvariable = crevar, state='readonly',font =('arial', 15, 'normal'))
        self.credits['values'] = ("Credits","Harshit","Roshan","Rupali","Sarthak","Sumit")
        self.credits.current(0)
        self.credits.place(x=20,y=450,width = 180)

        self.credits.bind("<<ComboboxSelected>>", credit)

        

maingui = tk.Tk()
maingui.iconbitmap("gi\icon.ico")
obj = win(maingui)
maingui.mainloop()