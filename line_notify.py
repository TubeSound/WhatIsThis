# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:35:10 2024

@author: docs9


"""

import requests


class LineNotify(object):
    
    API_URL = 'https://notify-api.line.me/api/notify'
    token =''
    
    def __init__(self):
        self.__headers = {'Authorization': 'Bearer ' + self.token}

    def send(self, message, image=None, sticker_package_id=None, sticker_id=None):
        payload = {
            'message': message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
        }
        files = {}
        if image != None:
            files = {'imageFile': open(image, 'rb')}
            
        r = requests.post(
            self.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
        )
        
def test():       
    notify = LineNotify()
    notify.send('Tensai')