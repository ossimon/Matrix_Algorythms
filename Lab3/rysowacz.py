from PIL import Image, ImageColor


def draw_rect(size: tuple):
    return Image.new('RGB', size)


def color_pixel(rectangle: Image, position: tuple, color_name: str):
    rectangle.putpixel(position, ImageColor.getrgb(color_name))


def draw_matrix(matrix, name='compressed_matrix.png'):
    img = draw_rect((len(matrix), len(matrix)))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 0:
                color = 'white'
            else:
                color = 'black'
            color_pixel(img, (i, j), color)

    img.save(name)


# img = draw_rect((10, 10))
# color_pixel(img, (0, 0), 'red')
# color_pixel(img, (0, 1), 'orange')
# color_pixel(img, (0, 2), 'yellow')
# color_pixel(img, (0, 3), 'green')
# color_pixel(img, (0, 4), 'cyan')
# color_pixel(img, (0, 5), 'blue')
# color_pixel(img, (0, 6), 'purple')
#
# img.save('sqr.png')

# draw_matrix(matrix)
