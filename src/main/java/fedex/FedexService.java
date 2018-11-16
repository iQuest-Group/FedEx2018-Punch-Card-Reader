package fedex;

import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author paul.stoia
 */
public class FedexService {

  public static void main(String[] args) throws IOException {
    String codes = new FedexService().processImage(FileUtils.readFileToByteArray(
            new File("C:\\Projects\\OpenCV\\DetectRectangles\\src\\main\\resources\\images\\fedex.png")));
    System.out.println();
  }

  public String processImage(byte[] sourceImage) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    Mat srcImage = Imgcodecs.imdecode(new MatOfByte(sourceImage), Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);

    Mat greyImg = convertToGreyscale(srcImage);

    Mat binaryImg = binaries(greyImg);

    Mat erodedImage = erode(binaryImg);

    Mat dilatedImage = dilate(erodedImage);

    List<MatOfPoint> allContours = findContours(dilatedImage);

    List<String> strings = new ArrayList<>();
    List<List<MatOfPoint>> clustersOfContours = findClustersOfContours(allContours);
    for (List<MatOfPoint> cluster : clustersOfContours) {
      List<Point> centersOfClusters = cluster.stream()
          .map(regionPoints -> computeCenterOfRegion(regionPoints.toList()))
          .collect(Collectors.toList());

      List<Point> sortedCenters = centersOfClusters.stream()
          .sorted(Comparator.comparingDouble(center -> center.y * 65_535 + center.x))
          .collect(Collectors.toList());

      double xMinBetweenNearestCenters = findXMinBetweenCenters(sortedCenters);

      List<List<Point>> linesOfCenters = splitCentersIntoLines(sortedCenters);

      String clusterString = linesOfCenters.stream()
          .map(line -> analyzeLine(line, xMinBetweenNearestCenters))
          .collect(Collectors.joining(" "));
      strings.add(clusterString);
    }
    return strings.stream().collect(Collectors.joining("\n"));
  }

  private String analyzeLine(List<Point> centerPoints, double xDistance) {
    StringBuilder asciiCode = new StringBuilder("1");
    centerPoints = centerPoints.stream()
        .sorted(Comparator.comparingDouble(center -> center.x))
        .collect(Collectors.toList());
    for (int i = 0; i < centerPoints.size() - 1; i++) {
      int noZero = (int) ((centerPoints.get(i + 1).x - centerPoints.get(i).x) / xDistance) - 1;
      if (noZero > 0) {
        asciiCode.append("0000000000000000".substring(0, noZero));
      }
      asciiCode.append("1");
    }
    return convertToChar(asciiCode.toString());
  }

  private String convertToChar(String binaryValue) {
    String result1 = "", result2 = "", result3 = "";
    String bigLetter = "0" + binaryValue + "00000000";
    bigLetter = bigLetter.substring(0, 8);
    if (Integer.parseInt(bigLetter, 2) >= 65 && Integer.parseInt(bigLetter, 2) <= 90) {
      result1 = String.valueOf((char) Integer.parseInt(bigLetter, 2));
    }
    String lowLetter = "0" + binaryValue + "00000000";
    lowLetter = lowLetter.substring(0, 8);
    if (Integer.parseInt(lowLetter, 2) >= 97 && Integer.parseInt(lowLetter, 2) <= 122) {
      result2 = String.valueOf((char) Integer.parseInt(lowLetter, 2));
    }
    String digit = "00" + binaryValue + "00000000";
    digit = digit.substring(0, 8);
    if (Integer.parseInt(digit, 2) >= 48 && Integer.parseInt(digit, 2) <= 57) {
      result3 = String.valueOf((char) Integer.parseInt(digit, 2));
    }
    return Stream.of(result1, result2, result3)
        .filter(result -> !result.equals(""))
        .collect(Collectors.joining("/"));
  }

  private List<List<Point>> splitCentersIntoLines(List<Point> centers) {
    List<List<Point>> linesOfCenters = new ArrayList<>();
    List<Point> currentLine = new ArrayList<>();
    double currentLineY = centers.get(0).y;
    double thresholdPercentage = 0.05 * currentLineY; // = 10 % * currentLineY
    for (Point center : centers) {
      if (Math.abs(currentLineY - center.y) < thresholdPercentage) {
        currentLine.add(center);
      } else {
        linesOfCenters.add(currentLine);
        currentLineY = center.y;
        thresholdPercentage = 0.05 * currentLineY;
        currentLine = new ArrayList<>();
        currentLine.add(center);
      }
    }
    linesOfCenters.add(currentLine);
    return linesOfCenters;
  }

  private double findXMinBetweenCenters(List<Point> xCoordsOfCenters) {
    double xMin = Double.MAX_VALUE;
    for (int i = 0; i < xCoordsOfCenters.size() - 1; i++) {
      double xDistance = xCoordsOfCenters.get(i + 1).x - xCoordsOfCenters.get(i).x;
      if (xDistance > 0 && xDistance < xMin) {
        xMin = xDistance;
      }
    }
    return xMin;
  }

  private Mat convertToGreyscale(Mat srcImage) {
    Mat greyscaleImage = srcImage.clone();
    Imgproc.cvtColor(srcImage, greyscaleImage, Imgproc.COLOR_RGB2GRAY);
    return greyscaleImage;
  }

  private Mat binaries(Mat srcImage) {
    Mat binaryImage = srcImage.clone();
    Imgproc.threshold(srcImage, binaryImage, 0, 255, Imgproc.THRESH_OTSU);
    return binaryImage;
  }

  private Mat erode(Mat srcImage) {
    Mat erodedImage = srcImage.clone();
    int erosion_size = 1;
    Mat
        structuringElement =
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosion_size + 1, 2 * erosion_size + 1));
    Imgproc.erode(srcImage, erodedImage, structuringElement);
    return erodedImage;
  }

  private Mat dilate(Mat srcImage) {
    Mat dilatedImage = srcImage.clone();
    int erosion_size = 5;
    Mat
        structuringElement =
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2 * erosion_size + 1, 2 * erosion_size + 1));
    Imgproc.dilate(srcImage, dilatedImage, structuringElement);
    return dilatedImage;
  }

  private List<MatOfPoint> findContours(Mat srcImage) {
    List<MatOfPoint> contours = new ArrayList<>();
    Imgproc.findContours(srcImage, contours, new Mat(),
                         Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);
    return contours.stream()
        .sorted(Comparator.comparingDouble(Imgproc::contourArea))
        .collect(Collectors.toList());
  }

  private List<List<MatOfPoint>> findClustersOfContours(List<MatOfPoint> contours) {
    List<List<MatOfPoint>> clustersOfContours = new ArrayList<>();

    int clusterStartIndex = 0;
    double firstContourArea = Imgproc.contourArea(contours.get(0));
    for (int index = 1; index < contours.size(); index++) {
      double secondContourArea = Imgproc.contourArea(contours.get(index));
      double diffPercentage = (secondContourArea - firstContourArea) * 100.0 / firstContourArea;
      if (diffPercentage < 5.0) {
        firstContourArea = secondContourArea;
      } else {
        List<MatOfPoint> newCluster = contours.subList(clusterStartIndex, index);
        clustersOfContours.add(newCluster);
        clusterStartIndex = index + 1;
      }
    }
    if (clusterStartIndex < contours.size()) {
      List<MatOfPoint> newCluster = contours.subList(clusterStartIndex, contours.size() - 1);
      clustersOfContours.add(newCluster);
    }
    return clustersOfContours.stream()
        .filter(cluster -> 25 <= cluster.size() && cluster.size() <= 30)
        .collect(Collectors.toList());
  }

  private Point computeCenterOfRegion(List<Point> regionPoints) {
    DoubleSummaryStatistics xSummaryStatistics = regionPoints.stream()
        .mapToDouble(point -> point.x)
        .summaryStatistics();
    DoubleSummaryStatistics ySummaryStatistics = regionPoints.stream()
        .mapToDouble(point -> point.y)
        .summaryStatistics();
    double xMin = xSummaryStatistics.getMin();
    double xMax = xSummaryStatistics.getMax();
    double yMin = ySummaryStatistics.getMin();
    double yMax = ySummaryStatistics.getMax();
    double centerX = xMin + (xMax - xMin) / 2.0;
    double centerY = yMin + (yMax - yMin) / 2.0;

    return new Point(centerX, centerY);
  }
}
