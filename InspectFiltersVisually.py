import tkinter as tk
from tkinter import ttk

import xarray as xr
import numpy as np
import cmocean.cm as cmo
import cartopy.crs as ccrs
import os

import sys
sys.path.append('./functions')
import filters

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  NavigationToolbar2Tk) 

class App:
    def __init__(self, root):
        self.root = root
    
        #defining sections
        #self.left = tk.Frame(root, width=200, height=400)
        #self.left.grid(row=0, column=0)
        
        #self.right= tk.Frame(root, width=1000, height=400,background='white')
        #self.right.grid(row=0, column=1)
        self.frame = tk.Frame(root, width=1200, height=800)
        self.frame.pack()

        # loading directory:
        tk.Label(self.frame, text='Directory:').pack()
        self.directory = tk.Entry(self.frame, width = 25)
        self.directory.pack()
        self.directory.bind('<Return>',self.locateFiles)
        
        self.nextb = tk.Button(self.frame, text = 'Next', command = self.load, width=10)
        self.nextb.pack()

        self.canvas = None  # Add reference to canvas
        self.toolbar = None  # Add reference to toolbar

    def locateFiles(self, event):
        self.dir = fr'{format(self.directory.get())}'
        if os.path.isdir(self.dir):
            self.dirlist = os.listdir(self.dir)
            #self.idx = -1
            self.idx = -1
            self.load()
            

    def load(self):
        # loading data and setting options for lat, lon and var
        self.idx +=1
        file = os.path.join(self.dir, self.dirlist[self.idx])
        self.ds = xr.load_dataset(file)
        Tb = self.ds['Brightness_temperature'].as_numpy()

        #filtering Tb
        Tb_filtered = np.copy(Tb)

        if filters.acceptanceFilter(Tb_filtered):
            Tb_filtered*=np.nan
        else:
            filters.swathFilter(Tb_filtered)

        #creating figure
        fig = Figure(figsize = (18,5), dpi=100,layout='constrained')
        ax = fig.add_subplot(2,1,1)
        #fig, ax = plot.subplots(2,1, sharex=True)
        ax.imshow(Tb.T, vmin=110, vmax=300)
        ax.set_title(self.dirlist[self.idx])
        ax2 = fig.add_subplot(2,1,2)
        ax2.imshow(Tb_filtered.T, vmin=110, vmax=300)

        # Destroy the previous canvas and toolbar if they exist
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()

        # making canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.canvas.draw()
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()




    def run(self):
        self.root.mainloop()



root = tk.Tk()
root.title('Filter Visualizer')
root.geometry("1800x1000")
app = App(root)
app.run()