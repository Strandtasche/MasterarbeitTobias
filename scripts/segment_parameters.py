# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:26:24 2018

@author: ma
"""

class SegmentParameters:
    
    class Parameters:
        def __init__(self, h_lower = 0, h_upper = 255, s_lower = 0, s_upper = 255, b_lower = 0, b_upper = 255, min_area = 100):
            self.h_lower = h_lower
            self.h_upper = h_upper
            self.s_lower = s_lower
            self.s_upper = s_upper
            self.b_lower = b_lower
            self.b_upper = b_upper
            self.min_area = min_area
    
    def __init__(self):
        self.plates = self.Parameters(5, 20, 38, 255, 0, 255, 3000)
        self.pfeffer = self.Parameters(10, 140, int(0.15*255), 255, int(0.2*255), int(0.6 * 255), 1000)
        self.kugel = self.Parameters(-40, 140, int(0.15*255), 255, int(0.2*255), int(0.6 * 255), 1000)
        self.weizen = self.Parameters(0, 255, int(0.15*255), 255, int(0.2*255), int(0.6 * 255), 1000)
        self.zylinder = self.Parameters(int(60/360 * 255), int(200/360 * 255), int(0.15*255), 255, int(0.2*255), int(0.6 * 255), 1000)
        
    
    def get_parameters(self, dataset):
        if 'quader' in dataset:
            return self.plates
        if 'Pfeffer' in dataset:
            return self.pfeffer
        if 'kugel' in dataset:
            return self.kugel
        if 'weizen' in dataset:
            return self.weizen
        if 'zylinder' in dataset:
            return self.zylinder
        # return self.pfeffer
        
        raise ValueError('Unknown dataset type: ' + dataset)