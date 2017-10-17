"""
Programmer: Chris Tralie
Purpose: My own animator of MOCAP data using OpenGL
"""
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import wx
from wx import glcanvas

from Geom3D.Cameras3D import *
from Geom3D.MeshCanvas import *
from Geom3D.Primitives3D import BBox3D
from sys import exit, argv
import random
import numpy as np
import scipy.io as sio
from pylab import cm
import os
import subprocess
import math
import time
from Skeleton import *

CENTER_ON_OBJECT = False
#If true, keep centering on bounding box around the object
#If false, stay in world coordinates

class SkeletonViewerCanvas(BasicMeshCanvas):
    def __init__(self, parent):
        BasicMeshCanvas.__init__(self, parent)    
        
        #Skeleton animation variables
        self.skeleton = Skeleton()
        self.animator = SkeletonAnimator(self.skeleton)
        self.animationState = 0
        self.animating = False
    
    def startAnimation(self, evt):
        self.animationState = 0
        self.animating = True
        self.Refresh()

    def repaint(self):
        X = self.animator.getState(self.animationState)
        if X.size > 0 and CENTER_ON_OBJECT:
            self.bbox = BBox3D()
            self.bbox.fromPoints(X)
            self.camera.centerOnBBox(self.bbox, 0, math.pi/2)
    
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        farDist = (self.camera.eye - self.bbox.getCenter()).flatten()
        farDist = np.sqrt(np.sum(farDist**2))
        farDist += self.bbox.getDiagLength()
        nearDist = farDist/50.0
        gluPerspective(180.0*self.camera.yfov/np.pi, float(self.size.x)/self.size.y, nearDist, farDist)
        
        #Set up modelview matrix
        self.camera.gotoCameraFrame()    
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [3.0, 4.0, 5.0, 0.0]);
        glLightfv(GL_LIGHT1, GL_POSITION,  [-3.0, -2.0, -3.0, 0.0]);
        
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0]);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        
        if self.animator:
            glDisable(GL_LIGHTING)
            self.animator.renderState(self.animationState)
        
        if self.animating:
            self.animationState = self.animationState + 1
            if self.animationState >= self.animator.NStates:
                self.animationState = self.animator.NStates - 1
                self.animating = False
            saveImageGL(self, "MOCAP%i.png"%self.animationState)
            
            self.Refresh()
        
        self.SwapBuffers()
    
    def initGL(self):        
        glutInit('')
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_LIGHT1)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

    def handleMouseStuff(self, x, y):
        #Invert y from what the window manager says
        y = self.size.height - y
        self.MousePos = [x, y]

    def MouseDown(self, evt):
        x, y = evt.GetPosition()
        self.CaptureMouse()
        self.handleMouseStuff(x, y)
        self.Refresh()
    
    def MouseUp(self, evt):
        x, y = evt.GetPosition()
        self.handleMouseStuff(x, y)
        self.ReleaseMouse()
        self.Refresh()

    def MouseMotion(self, evt):
        x, y = evt.GetPosition()
        [lastX, lastY] = self.MousePos
        self.handleMouseStuff(x, y)
        dX = self.MousePos[0] - lastX
        dY = self.MousePos[1] - lastY
        if evt.Dragging():
            if evt.MiddleIsDown():
                self.camera.translate(dX, dY)
            elif evt.RightIsDown():
                self.camera.zoom(-dY)#Want to zoom in as the mouse goes up
            elif evt.LeftIsDown():
                self.camera.orbitLeftRight(dX)
                self.camera.orbitUpDown(dY)
        self.Refresh()

class SkeletonViewerFrame(wx.Frame):
    (ID_LOADSKELETON_AMC, ID_LOADSKELETON_ASF, ID_SAVESCREENSHOT) = (1, 2, 3)
    
    def __init__(self, parent, id, title, pos=DEFAULT_POS, size=DEFAULT_SIZE, style=wx.DEFAULT_FRAME_STYLE, name = 'GLWindow', mesh1 = None, mesh2 = None):
        style = style | wx.NO_FULL_REPAINT_ON_RESIZE
        super(SkeletonViewerFrame, self).__init__(parent, id, title, pos, size, style, name)
        #Initialize the menu
        self.CreateStatusBar()
        
        self.size = size
        self.pos = pos
        
        self.asffilename = ''
        
        filemenu = wx.Menu()
        menuOpenASF = filemenu.Append(SkeletonViewerFrame.ID_LOADSKELETON_ASF, "&Load ASF File","Load ASF File")
        self.Bind(wx.EVT_MENU, self.OnLoadASFFile, menuOpenASF)
        menuOpenAMC = filemenu.Append(SkeletonViewerFrame.ID_LOADSKELETON_AMC, "&Load AMC File","Load AMC File")
        self.Bind(wx.EVT_MENU, self.OnLoadAMCFile, menuOpenAMC)
        menuSaveScreenshot = filemenu.Append(SkeletonViewerFrame.ID_SAVESCREENSHOT, "&Save Screenshot", "Save a screenshot of the GL Canvas")
        self.Bind(wx.EVT_MENU, self.OnSaveScreenshot, menuSaveScreenshot)
        menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
        
        # Creating the menubar.
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
        self.glcanvas = SkeletonViewerCanvas(self)
        
        self.rightPanel = wx.BoxSizer(wx.VERTICAL)
        
        #Buttons to go to a default view
        animatePanel = wx.BoxSizer(wx.HORIZONTAL)
        animateButton = wx.Button(self, -1, "Animate")
        self.Bind(wx.EVT_BUTTON, self.glcanvas.startAnimation, animateButton)
        animatePanel.Add(animateButton, 0, wx.EXPAND)
        self.rightPanel.Add(wx.StaticText(self, label="Animation Options"), 0, wx.EXPAND)
        self.rightPanel.Add(animatePanel, 0, wx.EXPAND)
        
        
        #Finally add the two main panels to the sizer        
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.glcanvas, 2, wx.EXPAND)
        self.sizer.Add(self.rightPanel, 0, wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Layout()
        self.Show()
    
    def OnLoadASFFile(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            self.asffilename = filepath
            self.glcanvas.skeleton = Skeleton()
            self.glcanvas.skeleton.initFromFile(filepath)
            self.glcanvas.Refresh()
        dlg.Destroy()
        return

    def OnLoadAMCFile(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            print(filepath)
            self.glcanvas.animator = SkeletonAnimator(self.glcanvas.skeleton)
            #self.glcanvas.animator.initFromFile(filepath)
            self.glcanvas.animator.initFromFileUsingOctave(self.asffilename, filepath)
            self.glcanvas.Refresh()
        dlg.Destroy()
        self.glcanvas.bbox = BBox3D()
        self.glcanvas.bbox.b = self.glcanvas.animator.getBBox()
        print("BBox = %s"%self.glcanvas.bbox)
        self.glcanvas.camera.centerOnBBox(self.glcanvas.bbox, math.pi/2, math.pi/2)
        return

    def OnSaveScreenshot(self, evt):
        dlg = wx.FileDialog(self, "Choose a file", ".", "", "*", wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            saveImageGL(self.glcanvas, filepath)
        dlg.Destroy()
        return

    def OnExit(self, evt):
        self.Close(True)
        return

class SkeletonViewer(object):
    def __init__(self, m1 = None, m2 = None):
        app = wx.App()
        frame = SkeletonViewerFrame(None, -1, 'SkeletonViewer', mesh1 = m1, mesh2 = m2)
        frame.Show(True)
        app.MainLoop()
        app.Destroy()

if __name__ == '__main__':
    m1 = None
    m2 = None
    if len(argv) >= 3:
        m1 = LaplacianMesh()
        m1.loadFile(argv[1])
        m2 = LaplacianMesh()
        m2.loadFile(argv[2])
    viewer = SkeletonViewer(m1, m2)
