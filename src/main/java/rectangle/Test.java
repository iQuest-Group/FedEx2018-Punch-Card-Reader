package rectangle;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @author paul.stoia
 */
public class Test {
  public static void main(String[] args) {
    //Loading the OpenCV core library
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    //Reading the Image from the file
    String inputFile = "C:/Projects/OpenCV/DetectRectangles/src/main/resources/images/rectangles.jpg";
    Mat srcImage = Imgcodecs.imread(inputFile);
    System.out.println("Image Loaded");

    String outputFile = "C:/Projects/OpenCV/DetectRectangles/src/main/resources/images/rectanglesDone.jpg";
    Imgcodecs.imwrite(outputFile, srcImage);
    System.out.println("Image written");
  }
}
