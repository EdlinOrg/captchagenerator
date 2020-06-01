import numpy as np

import PILasOPENCV


# getmask in PILasOPENCV does not work with certain characters / fonts / sizes
# this is a quick fix
# XXX: fix properly and make PR to PILasOPENCV
def getmaskFix(text, ttf_font):
    slot = ttf_font.glyph
    width, height, baseline = PILasOPENCV.getsize(text, ttf_font)
    Z = np.zeros((height, width), dtype=np.ubyte)
    x, y = 0, 0
    previous = 0
    for c in text:
        ttf_font.load_char(c)
        bitmap = slot.bitmap
        top = slot.bitmap_top
        left = slot.bitmap_left
        w,h = bitmap.width, bitmap.rows

        #My modification
        if previous == 0 and (w != width or h != height):
            Z = np.zeros((h, w), dtype=np.ubyte)
            y = 0
        else:
            y = height-baseline-top
        if y<=0: y=0
        kerning = ttf_font.get_kerning(previous, c)
        x += (kerning.x >> 6)
        character = np.array(bitmap.buffer, dtype='uint8').reshape(h,w)
#        try:
        Z[y:y+h,x:x+w] += character
#        except ValueError:
#            while x+w>Z.shape[1]:
#                x = x - 1
#            print("new", x, y, w, h, character.shape, type(bitmap))
#            if x>0:
#                Z[:character.shape[0],x:x+w] += character
        x += (slot.advance.x >> 6)
        previous = c
    return Z
