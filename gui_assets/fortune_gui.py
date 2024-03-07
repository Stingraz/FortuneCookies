# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:49:09 2024
@author: Michelle Fribance

The purpose of this script is to create a GUI to present the user with a text
prompt and a button to request a fortune from one of the pretrained Markov Chains 
models (pickle files) when the program is run. The pickle files, 
Markov_fortune_generator.py, and this script should all be in the same
folder in your local system. Adjust the path on line 17 before running script.

Uses customtkinter: https://customtkinter.tomschimansky.com/
"""
import os

import pandas as pd
import random
from tkinter import messagebox, StringVar, PhotoImage
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from Markov_fortune_generator import generate_fortune 
#from lstm_fortune_generator import generate_sentence

# --------------- Create and configure the main app window ------------------ #

#app = ctk.CTkToplevel()
app = tk.Toplevel()
app.title("The Cookie Oracle")
app.geometry("1116x697")
app.iconbitmap("cookie_icon.ico")  # Sets the icon for the upper left corner and taskbar

# Set the app and component colors:
ctk.set_appearance_mode("dark")  # "system", "light", "dark"
ctk.set_default_color_theme("blue") 
   

# Configure the app grid to be 2 columns wide and 6 rows high
app.grid_columnconfigure(0, weight=0)
app.grid_columnconfigure(1, weight=1)  # right column will be twice as wide

# Set row weights to allocate more space for the textbox frame
app.grid_rowconfigure(0, weight=0)  
app.grid_rowconfigure(1, weight=0)  
app.grid_rowconfigure(2, weight=0)  
app.grid_rowconfigure(3, weight=1)  # Row 3 will be twice as tall as the others
app.grid_rowconfigure(4, weight=0)  

background_image = PhotoImage(file="mountain_temple_1.png")
# Show image using label 
background_label = ctk.CTkLabel(app, text="", image=background_image) 
background_label.place(x=0, y=0, relwidth=1, relheight=1) # Must use place for background, so you can put other stuff overtop

# Create and configure the label for instruction
#label = customtkinter.CTkLabel(app, text="Click the button to generate a fortune!", font=("Arial", 14))
#label.grid(row=0, column=0) # Specify the grid location


# -------- Create text entry box for setting fortune topic for LSTM ---------- #

# Create and configure the frame for fortune topic selection
topic_frame = ctk.CTkFrame(app)
topic_frame.grid(row=1, column=0, padx=10, pady=(10,10), sticky="ew") #, rowspan=2) 
topic_frame.grid_remove()  # Hide the frame initially

# Create and configure the label for fortune topic
topic_label = ctk.CTkLabel(topic_frame, text="LSTM fortune topic?", font=("Arial", 12))
topic_label.grid(row=0, column=0)

def entry_callback(event):
    print("Entered Topic:", topic_entry.get())
    
# Initialize StringVar to track the selected fortune topic
entered_topic = StringVar(value="Enter a fortune topic")

# Function to handle fortune topic entry
def handle_entry(event):
    entered_topic.set(topic_entry.get())
    entry_callback(event)

# Create and configure the entry box for fortune topic entry
topic_entry = ctk.CTkEntry(topic_frame, textvariable=entered_topic)
topic_entry.grid(row=1, column=0, columnspan=3)
topic_entry.bind("<Return>", handle_entry)  # Bind return key to handle entry


# ------ Create sliders for setting state size if Markov model selected ----- #

# Define default parameters to determine which pretrained Markov model to use:
state_size = 1  

slider_frame = ctk.CTkFrame(app)
slider_frame.grid(row=1, column=0, padx=10, pady=(10,10), sticky="ew") 
slider_frame.grid_remove()  # Hide the slider frame initially

# Create and configure the title label for the slider
slider_label = ctk.CTkLabel(slider_frame, text="Markov Model:", font=("Arial", 12))
slider_label.grid(row=0, column=1, sticky="ew") 

def slider_event(value):
    global state_size  # Access the global state_size variable
    state_size = int(value)   # Update the value of the state_size, which changes which Markov model to use
    print("Slider adjusted, switching to pretrained Markov model with state size", state_size)

# Initialize IntVar for the slider
slider_var = tk.IntVar(value=state_size)
slider = ctk.CTkSlider(slider_frame, from_=1, to=4, number_of_steps=3, 
                                 command=slider_event, variable=slider_var)

slider.grid(row=1, column=1, sticky="ew") 

# Create and configure the end point labels for the slider
slider_label_left = ctk.CTkLabel(slider_frame, text="More diversity", font=("Arial", 12))
slider_label_left.grid(row=2, column=0, sticky="s") 

slider_label_right = ctk.CTkLabel(slider_frame, text="More coherence", font=("Arial", 12))
slider_label_right.grid(row=2, column=2, sticky="s") 


# ----------- Create radio buttons for setting the model type --------------- #

# Create and configure the frame for model selection
model_frame = ctk.CTkFrame(app)
model_frame.grid(row=0, column=0, padx=100, pady=(10,10), sticky="ew")

# Create and configure the label for model type
model_label = ctk.CTkLabel(model_frame, text="Cookie Flavour:", font=("Arial", 12, "bold"))
model_label.grid(row=0, column=0, sticky="w")

# Initialize IntVar to track the selected model type
selected_model = StringVar(value="")

# Function to handle radio button selection
def select_model(value):
    selected_model.set(value)
    print("Selected Model:", selected_model.get())
    fortune_label.configure(text="")
    background_label.configure(image=background_image)
    background_label.image = background_image  # Keep a reference to prevent garbage collection
    # Show the cookie frame if a radio button is selected
    cookie_frame.grid(row=2, column=0, padx=10, pady=(10,10)) #, sticky="ew")
    
    # Show the fortune topic frame if LSTM is selected, hide it otherwise
    #if value == "LSTM":
    #    topic_frame.grid(row=1, column=0, padx=100, pady=(10,10)) #, sticky="ew")
    #else:
    #    topic_frame.grid_remove()
    
    # Show the slider frame if Markov Chains is selected, hide it otherwise
    if value == "Markov Chains":
        slider_frame.grid(row=1, column=0, padx=10, pady=(10,10)) #, sticky="ew")
    else:
        slider_frame.grid_remove()
        
# Create and configure radio buttons for different model types
radiobutton_1 = ctk.CTkRadioButton(model_frame, text="Markov Chains", variable=selected_model, value="Markov Chains", 
                                             command=lambda: select_model("Markov Chains"))
radiobutton_1.grid(row=1, column=0, sticky="w")

radiobutton_2 = ctk.CTkRadioButton(model_frame, text="Word n-grams", variable=selected_model, value="Word n-grams", 
                                             command=lambda: select_model("Word n-grams"))
radiobutton_2.grid(row=2, column=0, sticky="w")

radiobutton_3 = ctk.CTkRadioButton(model_frame, text="LSTM", variable=selected_model, value="LSTM", 
                                             command=lambda: select_model("LSTM"))
radiobutton_3.grid(row=3, column=0, sticky="w")

radiobutton_4 = ctk.CTkRadioButton(model_frame, text="RNN", variable=selected_model, value="RNN", 
                                             command=lambda: select_model("RNN"))
radiobutton_4.grid(row=4, column=0, sticky="w")

radiobutton_5 = ctk.CTkRadioButton(model_frame, text="GPT-2", variable=selected_model, value="GPT-2", 
                                             command=lambda: select_model("GPT-2"))
radiobutton_5.grid(row=5, column=0, sticky="w")


# -------- Create and configure the button to generate a new fortune -------- #

def generate_new_fortune():
    """elif selected_model.get() == "LSTM":
            # Get the selected topic from the LSTM combobox (dropdown)
            seed_word = entered_topic.get()
            fortune = generate_sentence(model_path='pretrained_models/lstmTextGeneration.pkl', 
                                        char_index_path='pretrained_models/char_index.pkl', 
                                        seed_word=seed_word, temperature=1.0, SEQLEN=20)"""    
    try:
        if selected_model.get() == "Markov Chains":
            fortune = generate_fortune(state_size)
        elif selected_model.get() == "Word n-grams":
            generated_fortunes = pd.read_csv(os.path.join('..', 'datasets', 'n-gram_generated_fortunes_100.csv'))
            fortune = random.choice(generated_fortunes['fortunes'])
        elif selected_model.get() == "GPT-2":
            generated_fortunes = pd.read_csv(os.path.join('..', 'datasets', 'GPT2_generated_fortunes_100-modular-early-stopping.csv'))
            fortune = random.choice(generated_fortunes['fortunes'])
        elif selected_model.get() == "LSTM":
            generated_fortunes = pd.read_csv(os.path.join('..', 'datasets', 'lstm_generated_fortunes_100.csv'))
            fortune = random.choice(generated_fortunes['fortunes'])
        elif selected_model.get() == "RNN":
            generated_fortunes = pd.read_csv(os.path.join('..', 'datasets', 'rnn_generated_fortunes_100.csv.csv'))
            fortune = random.choice(generated_fortunes['fortunes'])
        else:
            # Handle other model types if needed
            fortune = "Please select a model type first"
            
        fortune_label.configure(text=fortune)
        background_image_new = PhotoImage(file="mountain_temple_2.png")
        background_label.configure(image=background_image_new)
        background_label.image = background_image_new  # Keep a reference to prevent garbage collection
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create a frame to hold the image
cookie_frame = ctk.CTkFrame(app)
cookie_frame.grid(row=2, column=0, padx=10, pady=(10,10), sticky="ew")
cookie_frame.grid_remove()  # Hide the frame initially

# Load the paper image file for the cookie fortune shaped button
pil_image = Image.open("cookie_button.png")
tk_image = ImageTk.PhotoImage(pil_image) # Convert to tk image format so tkinter can use it

# Create a label to display the image inside the frame
cookie_label = ctk.CTkLabel(cookie_frame, text="", image=tk_image)
cookie_label.grid(row=0, column=0)

# Create an invisible button overtop of the cookie image
cookie_button = tk.Button(cookie_frame, text="Generate unique fortune", font=("Arial", 12, "italic"), command=generate_new_fortune, bd=0)
cookie_button.grid(row=0, column=0,  padx=100, sticky="e")


# -------- Create the label for displaying the generated fortune ------------ #

fortune_label = tk.Label(app, text="", font=("Rage Italic", 32), wraplength=600) 
#fortune_label = tk.Label(fortune_frame, text="", font=("Palace Script MT", 80), wraplength=600)
fortune_label.grid(row=3, column=1, pady=(0, 30), sticky="n") 

# Run the app
app.mainloop()
