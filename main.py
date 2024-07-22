import tkinter as tk
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter import ttk, StringVar, IntVar
from PIL import ImageTk, Image
from tkinter import messagebox
from PIL import Image
import create
import train
import absensi



def s_exit():
    exit(0)

def putwindow():

    window = Tk()
    window.geometry("800x500")
    window.configure(background='#1E1E1E')
    window.title("Attendance System")
    window.geometry("800x700")
    window.resizable(False,False)
    tkinter.Label(window, text = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -', bg = '#A7AEB5').pack(fill = 'x')
    tkinter.Label(window, text = "WELCOME TO AUTOMATIC ATTENDANCE", font = ("Poppins",20), fg = "black", bg = "#A7AEB5").pack(fill = "x")
    tkinter.Label(window, text = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -', bg = '#A7AEB5').pack(fill = 'x')

    tkinter.Label(window, text = "GUIDELINES TO USE THIS SOFTWARE", font = 18, fg = "black", bg = "#A7AEB5").pack(fill = "x")

    tkinter.Label(window, text = " ", bg = "#1E1E1E").pack(fill = 'y')

    tkinter.Label(window, text = ": \n\n"
                                 "This software allows user to:\n\n"
                                 "1) CREATE DATASET using MTCNN face detection and alignment\n"
                                 "2) TRAIN FaceNet for face recognition                     \n"
                                 "3) Do both                                                \n\n\n "
                                                   , fg = "#FFFFFF", bg = "#20262C").pack(fill = "y")

    tkinter.Label(window, text = "\n\n", bg = "#1E1E1E").pack(fill = 'y')


    tkinter.Label(window, text = ": \n\n"
                                 "**************   IMPORTANT   *************\n\n"
                                 "1) Ketika dalam proses CREAT data, pastikan kondisi cahaya baik, agar hasil yang didapatkan lebih baik\n"
                                 "   \n"
                                 "2) Klik 'q' untuk menutup halaman ketika sudah membuat Dataset baru, dan ingin membuat data baru.\n\n"
                                 "3) Pastikan menekan tombol keyboard pada gambar di window dan tidak di terminal                   \n"
                  , fg = "#FFFFFF", bg = "#20262C").pack(fill = "y")

    def cont_inue():
        window.destroy()
        show()

    btn1 = tkinter.Button(window, text = "CONTINUE", fg = "white", bg = '#343B42', command = cont_inue)
    btn1.place(x=360, y=550, width=80)

    window.mainloop()

def show():
    window2 = Tk()
    window2.title("Attendance System")
    window2.geometry("800x500")
    window2.configure(background='#1E1E1E')
    window2.resizable(False,False)
    tkinter.Label(window2, text = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -', bg = '#A7AEB5').pack(fill='x')
    tkinter.Label(window2, text = "WELCOME TO AUTOMATIC ATTENDANCE", font = ("Poppins",20), fg = "black", bg = "#A7AEB5").pack(fill="x")
    tkinter.Label(window2, text = '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -', bg = '#A7AEB5').pack(fill='x')

    tkinter.Label(window2, text = "\n\n ",bg = "#1E1E1E").pack(fill = 'y')

    tkinter.Label(window2, text = "Click 'TRAIN' untuk TRAIN Dataset yang telah dibuat.\n\n"
                                  "Click 'CREATE' untuk membuat Dataset baru.                     \n\n"
                                  "Click 'ABSENSI' untuk mulai ABSENSI mahasiswa  \n",
                  fg = "#FFFFFF", bg = "#20262C").pack(fill = 'y')
                  
    # bottom_frame = tkinter.Frame(window2).pack(side = "bottom")

    def train():
        print('Train Data')
        show_train()
        
    def create():
        print('Create Data')
        window2.destroy()
        show_create()
        
    def absensi():
        print('Absensi')
        show_absensi()


    btn1 = tkinter.Button(window2, text = "CREATE", fg = "white", bg = '#343B42', command = create)
    btn1.place(x=255, y=320, width=70)

    btn2 = tkinter.Button(window2, text = "TRAIN", fg = "white", bg = '#343B42', command = train)
    btn2.place(x=360, y=320, width=70)

    btn3 = tkinter.Button(window2, text = "ABSENSI", fg = "white", bg = '#343B42', command = absensi)
    btn3.place(x=470, y=320, width=70)

    btn4 = tkinter.Button(window2, text = "EXIT", fg = "black", bg = '#A6A6A7', command = s_exit)
    btn4.place(x=360, y=370, width=70)

    window2.mainloop()

def show_create():
    window3 = Tk()
    window3.title("Attendance System")
    window3.geometry("800x400")
    window3.configure(background='#1E1E1E')
    tkinter.Label(window3, text = "CREAT NEW DATASET", font = ("Poppins",20), fg = "black", bg = "#A7AEB5").pack(fill="x")
    tkinter.Label(window3, text = "\n\n ", bg = "black").pack(fill = 'y')

    tkinter.Label(window3, text = "\n\n ",bg = "black").pack(fill = 'y')

    tkinter.Label(window3, text = "Masukkan Nama : ", fg = "white", bg = "#20262C").place(x=100, y=70, width=600)
    inpNama = tkinter.Entry(window3)
    inpNama.place(x=100, y=90, width=600)

    tkinter.Label(window3, text = "Masukkan Nim : ",fg = "white", bg = "#20262C").place(x=100, y=120, width=600)
    inpNim = tkinter.Entry(window3)
    inpNim.place(x=100, y=140, width=600)

    tkinter.Label(window3, text = "Masukkan Kelas : ",fg = "white", bg = "#20262C").place(x=100, y=170, width=600)
    inpKelas = tkinter.Entry(window3)
    inpKelas.place(x=100, y=190, width=600)

    get_f = 0

    def submit():
        parameters = inpNama.get(), inpNim.get(), inpKelas.get()
        print(parameters)

        get_f = create.rekamWajah(parameters)

        if get_f == 1:
            tkinter.messagebox.showinfo("Attendance", "Dataset Created")
    
    def home():
        window3.destroy()
        gotohome()


    btn5 = tkinter.Button(window3, text = "CONTINUE", fg = "white", bg = '#343B42', command = submit)
    btn5.place(x=250, y=250, width=90)

    btn6 = tkinter.Button(window3, text = "HOME", fg = "white", bg = '#343B42', command = home)
    btn6.place(x=450, y= 250, width=90)


    window3.mainloop()

def show_train():
    window4 = Tk()
    train.train2()
    window4.mainloop()

def show_absensi():
    absensi.absen()

def gotohome():
    show()


if __name__ == "__main__":
    # show_create()
     putwindow()
