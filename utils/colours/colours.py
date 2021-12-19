
import random

# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]


# Distinct list of colours generated at: http://phrogz.net/css/distinct-colors.html

colours_raw = "#ff2929, #ff0000, #e50000, #d92323, #d90000, #bf0f0f, #b21d1d, #a60000, #991818, #ff5429, #f23000, #d93911, #bf3f1f, #b22400, #993218, #991f00, #ff7e29, #ff6600, #d95700, #bf5f1f, #a64200, #994c18, #ffa929, #ff9900, #d98200, #bf7f1f, #a6690d, #f2c200, #bf9f1f, #a68500, #ffff00, #f2f200, #d9d923, #bfbf1f, #999900, #ccff00, #bbe612, #9fbf1f, #87a60d, #99ff00, #91e612, #7fbf1f, #69a60d, #78f227, #5fbf1f, #4c9918, #54ff29, #2bd900, #26bf00, #2ca60d, #13f240, #1fbf3f, #1ba636, #00f261, #11d961, #1fbf5f, #1ba652, #00f291, #11d989, #0da669, #00ffcc, #11d9b1, #1fbf9f, #00997a, #00ffff, #1fbfbf, #009999, #00ccff, #0f9cbf, #007a99, #29a9ff, #0099ff, #2390d9, #0082d9, #0f79bf, #1b6ea6, #0063a6, #297eff, #0066ff, #1267e6, #236cd9, #0052cc, #0f56bf, #1b52a6, #0042a6, #2954ff, #0030f2, #123de6, #2347d9, #0029cc, #0f32bf, #0d2ca6, #183299, #0000f2, #0000e6, #0f0fbf, #4723d9, #361ba6, #2100a6, #6100f2, #7225e6, #560fbf, #9100f2, #9023d9, #790fbf, #6300a6, #c200f2, #b423d9, #951db3, #7a0099, #f200f2, #e600e6, #bf0fbf, #a61ba6, #ff29d4, #d911b1, #b31d95, #99187f, #ff29a9, #e5008a, #bf0f79, #a60063, #ff297e, #ff0066, #d91161, #bf004d, #b21d59, #a60042, #ff2954, #ff0033, #e5123d, #d92347, #cc0029, #bf0f32, #b21d3b, #a60021, #990c28"

colours = [[0,255,0]]
hex_colours = []
colours_raw = colours_raw.split(', ')
random.shuffle(colours_raw)
for colour in colours_raw:
    print(colour)
    if colour.startswith('#'):
        hex_colours.append(colour)
        colour = colour.strip('#')
        hex = list(int(colour[i:i+2], 16) for i in (0, 2, 4))
        hex.reverse()
        print(hex)
        colours.append(hex)


print(colours)
print(len(colours))