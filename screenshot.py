from tkinter import *
import pyautogui

import datetime


def take_bounded_screenshot(x1, y1, x2, y2):
    image = pyautogui.screenshot(region=(int(x1), int(y1), int(x2), int(y2)))
    file_name = datetime.datetime.now().strftime("%f")
    image.save("divine_font_gems.png")


class Application():
    def __init__(self, master):
        self.snip_surface = None
        self.master = master
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None

        self.master.geometry('400x50+200+200')  # set new geometry
        self.master.title('Lil Snippy')

        self.menu_frame = Frame(master)
        self.menu_frame.pack(fill=BOTH, expand=YES, padx=1, pady=1)

        self.buttonBar = Frame(self.menu_frame, bg="")
        self.buttonBar.pack()

        #self.snipButton = Button(self.buttonBar, width=5, height=5, command=self.create_screen_canvas, background="green")
        #self.snipButton.pack()

        self.master_screen = Toplevel(self.master)
        self.master_screen.withdraw()
        self.master_screen.attributes("-transparent", "maroon3")
        self.picture_frame = Frame(self.master_screen, background="maroon3")
        self.picture_frame.pack(fill=BOTH, expand=YES)

    def create_screen_canvas(self):
        self.master_screen.deiconify()
        self.master.withdraw()

        self.snip_surface = Canvas(self.picture_frame, cursor="cross", bg="grey11")
        self.snip_surface.pack(fill=BOTH, expand=YES)

        self.snip_surface.bind("<ButtonPress-1>", self.on_button_press)
        self.snip_surface.bind("<B1-Motion>", self.on_snip_drag)
        self.snip_surface.bind("<ButtonRelease-1>", self.on_button_release)

        self.master_screen.attributes('-fullscreen', True)
        self.master_screen.attributes('-alpha', .3)
        self.master_screen.lift()
        self.master_screen.attributes("-topmost", True)

    def on_button_release(self, event):
        self.display_rectangle_position()

        if self.start_x <= self.current_x and self.start_y <= self.current_y:
            #print("right down")
            take_bounded_screenshot(self.start_x, self.start_y, self.current_x - self.start_x, self.current_y - self.start_y)

        elif self.start_x >= self.current_x and self.start_y <= self.current_y:
            #print("left down")
            take_bounded_screenshot(self.current_x, self.start_y, self.start_x - self.current_x, self.current_y - self.start_y)

        elif self.start_x <= self.current_x and self.start_y >= self.current_y:
            #print("right up")
            take_bounded_screenshot(self.start_x, self.current_y, self.current_x - self.start_x, self.start_y - self.current_y)

        elif self.start_x >= self.current_x and self.start_y >= self.current_y:
            #print("left up")
            take_bounded_screenshot(self.current_x, self.current_y, self.start_x - self.current_x, self.start_y - self.current_y)

        self.exit_screenshot_mode()
        return event

    def exit_screenshot_mode(self):
        self.snip_surface.destroy()
        self.master_screen.withdraw()
        self.master.destroy()
        #root.deiconify()

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.snip_surface.canvasx(event.x)
        self.start_y = self.snip_surface.canvasy(event.y)
        self.snip_surface.create_rectangle(0, 0, 1, 1, outline='red', width=3, fill="maroon3")

    def on_snip_drag(self, event):
        self.current_x, self.current_y = (event.x, event.y)
        # expand rectangle as you drag the mouse
        self.snip_surface.coords(1, self.start_x, self.start_y, self.current_x, self.current_y)

    def display_rectangle_position(self):
        pass
        print(self.start_x)
        print(self.start_y)
        print(self.current_x)
        print(self.current_y)



def screenshot():
    root = Tk()
    app = Application(root)
    app.create_screen_canvas()
    root.mainloop()
    root.quit()
    