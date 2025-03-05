import numpy as np
import cv2
class EdgeDetection:
    
    @staticmethod
    def convolve(image: np.ndarray, gx: np.ndarray,gy: np.ndarray):
        if(len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(image.shape)
        rows, cols = image.shape
        kernel_rows, kernel_cols = gx.shape
        output_rows, output_cols = rows, cols
        if kernel_rows%2 == 0:
            output_rows -= 1
            output_cols -= 1
        else:
            output_rows -= 2
            output_cols -= 2
        i_x = np.zeros((output_rows, output_cols), dtype=np.float32)
        i_y = np.zeros((output_rows, output_cols), dtype=np.float32)

        for i in range(0,rows-kernel_rows+1):

            for j in range(0, cols-kernel_cols+1):
                square = image[i:i+kernel_rows, j:j+kernel_cols]
                i_x[i, j] = np.sum((square * gx))
                i_y[i, j] = np.sum((square * gy))
        

        return i_x, i_y

    
    @staticmethod
    def Sobel(image:np.ndarray,direction:str='mag',kSize:int=3):
        g_x, g_y = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]] ,dtype=np.float32), np.array([[-1, -2, -1],[ 0,  0,  0],[ 1,  2,  1]], dtype=np.float32)
        if kSize==5:
            g_x, g_y = (
            np.array([ 
            [-2, -1,  0,  1,  2],
            [-3, -2,  0,  2,  3],
            [-4, -3,  0,  3,  4],
            [-3, -2,  0,  2,  3],
            [-2, -1,  0,  1,  2]], dtype=np.float32),
            np.array([
            [-2, -3, -4, -3, -2],
            [-1, -2, -3, -2, -1],
            [ 0,  0,  0,  0,  0],
            [ 1,  2,  3,  2,  1],
            [ 2,  3,  4,  3,  2]], dtype=np.float32))
        elif kSize==7:
            g_x, g_y = np.array([[-3, -2, -1,  0,  1,  2,  3],[-4, -3, -2,  0,  2,  3,  4],[-5, -4, -3,  0,  3,  4,  5],[-6, -5, -4,  0,  4,  5,  6],[-5, -4, -3,  0,  3,  4,  5],[-4, -3, -2,  0,  2,  3,  4],[-3, -2, -1,  0,  1,  2,  3]], dtype=np.float32), np.array([[-3, -4, -5, -6, -5, -4, -3],[-2, -3, -4, -5, -4, -3, -2],[-1, -2, -3, -4, -3, -2, -1],[ 0,  0,  0,  0,  0,  0,  0],[ 1,  2,  3,  4,  3,  2,  1],[ 2,  3,  4,  5,  4,  3,  2],[ 3,  4,  5,  6,  5,  4,  3]], dtype=np.float32)
                
        i_x, i_y= EdgeDetection.convolve(image=image,gx=g_x,gy=g_y)
        if direction.lower() == 'x':
            print(f'x done')

            return i_x
        elif direction.lower() == 'y':
            print(f'y done')

            return i_y
        else:
            print(f'mag done')
            return np.sqrt(np.square(i_x) + np.square(i_y))
    @staticmethod
    def prewitt(image:np.ndarray,direction:str):
        g_x, g_y =np.array([[-1,  0,  1],[-1,  0,  1],[-1,  0,  1]],dtype=np.float32), np.array([[-1, -1, -1],[ 0,  0,  0],[ 1,  1,  1]],dtype=np.float32)

        i_x, i_y= EdgeDetection.convolve(image,g_x,g_y)
        if direction == 'x':
            cv2.imshow("i_x",i_x)
            cv2.waitKey(0)
            return i_x
        elif direction == 'y':
            cv2.imshow("i_y",i_y)
            cv2.waitKey(0)
            
            return i_y
        else:
            mag=np.sqrt(np.square(i_x) + np.square(i_y))
            cv2.imshow("mag",mag)
            cv2.waitKey(0)
            
            return mag
     
    @staticmethod
    def roberts(image:np.ndarray,direction:str):

        g_x, g_y = np.array([[1, 0], [0, -1]], dtype=np.float32), np.array([[0, 1], [-1, 0]], dtype=np.float32)
        i_x, i_y= EdgeDetection.convolve(image,g_x,g_y)
        if direction.lower() == 'x':
            return i_x
        elif direction.lower() == 'y':
            return i_y
        else:
            return np.sqrt(np.square(i_x) + np.square(i_y))
    @staticmethod
    def Canny(image:np.ndarray, ksize:tuple=(5,5), sigma:float=1.4, low_threshold:int=50, high_threshold:int=150):
        edges=cv2.GaussianBlur(image,ksize,sigma)
        edges = cv2.Canny(image, low_threshold, high_threshold)
        return edges