
import random

# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]


# Distinct list of colours generated at: http://phrogz.net/css/distinct-colors.html

colours_raw = "#00ccff, #66adff, #3b6594, #3b4bb8, #7258db, #a53dff, #4f0094, #a31db8, #db58c1, #ff298d, #940045, #ff667a, #943b47, #3b9fb8, #2c6db8, #3d57ff, #001494, #2500b8, #b866ff, #e014ff, #862f94, #b81d99, #ff66ad, #ff1434, #b82c3f, #1482ff, #004594, #667aff, #3300ff, #482cb8, #8449b8, #eb66ff, #ff14d0, #943b82, #b82c6d, #ff3d57, #940014"
colours_raw += ", #ff3429, #b80900, #94403b, #db5635, #ff7429, #b8531d, #ff9429, #b86a1d, #945c23, #b87700, #94753b, #b89300, #fff200, #eaff29, #86940c, #ff6e66, #b8413b, #ff5429, #b85f49, #ff8e52, #b87049, #ffa852, #b88149, #ffad14, #b88c3b, #ffd014, #b8a249, #fff766, #cedb58, #db3d35, #94130c, #ff8566, #943a23, #db4d00, #944b23, #db6e00, #944a00, #ffc252, #94640c, #ffe066, #94790c, #b8b13b, #a5b800"

colours = [[0,255,0]]
hex_colours = []
colours_raw = colours_raw.split(', ')
# random.shuffle(colours_raw)
for colour in colours_raw:
    print(colour)
    if colour.startswith('#'):
        hex_colours.append(colour)
        colour = colour.strip('#')
        hex = list(int(colour[i:i+2], 16) for i in (0, 2, 4))
        # while hex[1] >= 230 and hex[0] < 100 and hex[2] < 100:
        #     hex = [random.randint(0,255) for _ in range(3)]
        hex.reverse()
        # print(hex)
        colours.append(hex)


print(colours)
print(len(colours))