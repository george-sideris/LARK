import cv2
from PIL import Image

patternSize = (6, 8)  # number of squares in X, Y
squareLength = 32  # in mm
markerLength = 18  # in mm
dpi = 600  # cv2 outputs to png at 72 dpi, PIL at whatever is specified
filePath = f"CharucoBoards/{patternSize[0]}x{patternSize[1]}_square{squareLength}_marker{markerLength}.png"

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
board = cv2.aruco.CharucoBoard(patternSize, squareLength, markerLength, arucoDict)

imageSize = [squares*round(squareLength*dpi/25.4) for squares in patternSize]
BGRimage = board.generateImage(imageSize)
# cv2.imwrite(filePath, BGRimage)

RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
PILimage = Image.fromarray(RGBimage)
PILimage.save(filePath, dpi=(dpi,dpi))