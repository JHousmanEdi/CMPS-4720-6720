import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

import java.io.File;
import javax.imageio.ImageIO;
import org.apache.commons.io.FilenameUtils;
/**
 * Created by jason on 3/26/17.
 */
public class Image_reader {

    public int[][] getImagePaths() throws IOException {
        File dir = new File("/home/jason/Documents/CMPS-4720-6720/images");
        long numfiles = Files.list(Paths.get("/home/jason/Documents/CMPS-4720-6720/images")).count();
        int numfiles1 = (int) numfiles;
        int[][] dimensions = new int[numfiles1][2];
        File[] directoryListing = dir.listFiles();
        int incrementer = 0;
        if (directoryListing != null) {
            for (File images : directoryListing) {
                String absolutePath = images.getAbsolutePath();
                opencv_core.Mat image = imread(absolutePath, CV_8UC1);
                dimensions[incrementer][0] = image.rows();
                dimensions[incrementer][1] = image.cols();
                ++incrementer;
                image.release();


            }
        }

        return dimensions;

    }

    public int sort2d(int[][] array) throws IOException {
        File dir = new File("/home/jason/Documents/CMPS-4720-6720/images");
        long numfiles = Files.list(Paths.get("/home/jason/Documents/CMPS-4720-6720/images")).count();
        int numfiles1 = (int) numfiles;
        int minheight = array[0][0];
        int minwidth = array[0][1];
        System.out.println("Height" + minheight + "Width" + minwidth);
        int minind = 0;
        for (int i = 1; i < numfiles1; i++) {
            if (array[i][0] < minheight && array[i][1] < minwidth) {
                minheight = array[i][0];
                minwidth = array[i][1];
                System.out.println("Height" + minheight + "Width" + minwidth);
                minind = i;
            }
        }
        return minind;

    }
    private static BufferedImage resizeImage(BufferedImage originalImage, int type, int IMG_WIDTH, int IMG_HEIGHT) {
        BufferedImage resizedImage = new BufferedImage(IMG_WIDTH, IMG_HEIGHT, type);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, IMG_WIDTH, IMG_HEIGHT, null);
        g.dispose();

        return resizedImage;
    }

    public static void main(String[] args) throws Exception {
        //opencv_core.Mat image = imread("/home/jason/Documents/Image_Data/benchmark/saiapr_tc-12/00/images/25.jpg", CV_8UC1);
        //opencv_core.Mat imageTiff = imwrite(image, "25.tiff");

        Image_reader reader = new Image_reader();
        File dir = new File("/home/jason/Documents/CMPS-4720-6720/images");
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File images : directoryListing) {
                String absolutePath = images.getAbsolutePath();
                BufferedImage originalImage = ImageIO.read(new File(absolutePath));
                String basename = FilenameUtils.getBaseName(absolutePath);
                int type = originalImage.getType() == 0 ? BufferedImage.TYPE_INT_ARGB : originalImage.getType();
                BufferedImage resizedImagePng = resizeImage(originalImage, type, 250, 250);
                ImageIO.write(resizedImagePng, "png", new File("/home/jason/Documents/CMPS-4720-6720/ResizedImages/" + basename));
            }
        }
        /*// Make sure it was successfully loaded.
        if (image == null) {
            System.out.println("Image not found!");
            System.exit(1);
        }

        UByteRawIndexer sI = image.createIndexer();
        ArrayList<ArrayList<Integer>> images = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> image_pixel = new ArrayList<Integer>();

        for (int y = 0; y < image.rows(); y++) {

            for (int x = 0; x < image.cols(); x++) {
                image_pixel.add(sI.get(y,x));

            }
            images.add(image_pixel);
        }
        System.out.println(images.get(0));*/

    }
}
