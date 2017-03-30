

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb;

/**
 * Created by jason on 3/28/17.
 */
public class Person_Classifier {
    protected static int height = 250;
    protected static int width = 250;
    protected static final Logger log = LoggerFactory.getLogger(Person_Classifier.class);
    protected static int channels = 3;
    protected static int numLabels = 2;
    protected static int batchSize = 5000;
    protected static int numDatapoints = 20000;
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 50;
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;


    private void seperateByLabel()throws IOException {
        File hasPerson = new File("/home/jason/Documents/CMPS-4720-6720/Person/Person");
        File noPerson = new File("/home/jason/Documents/CMPS-4720-6720/Person/No_Person");
        ArrayList<String[]> lines = new ArrayList<String[]>();
        Scanner scanner = new Scanner(new File("/home/jason/Documents/CMPS-4720-6720/has_person_data_flat.csv"));
        scanner.useDelimiter(",");
        while(scanner.hasNextLine()){
            String line = scanner.nextLine();
            lines.add(line.split(","));
        }
        File dir = new File("/home/jason/Documents/CMPS-4720-6720/ResizedImages");
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File images : directoryListing) {
                String filename = images.getName();
                int filenumber = Integer.parseInt(filename);
                for(int i = 0; i < lines.size(); i++){
                    if(Integer.parseInt(lines.get(i)[0]) == filenumber){
                        if(Integer.parseInt(lines.get(i)[1]) == 1){
                            FileUtils.copyFileToDirectory(images, hasPerson);

                        }
                        if(Integer.parseInt(lines.get(i)[1]) == 0){
                            FileUtils.copyFileToDirectory(images, noPerson);
                        }

                    }
                }



            }
        }
    }
    public static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    public static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    public static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    public static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }


    public static void main(String[] args)throws Exception {
        Person_Classifier main = new Person_Classifier();
        Nd4j.create(1);

        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true)
                .setMaximumDeviceCache(2L*1024L*1024L*1024L)
                .allowCrossDeviceAccess(true);
        log.info("Processing and Loading data....");

        //ImageRecordReader recordReader = new ImageRecordReader(250, 250, 3);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File parentDir = new File(System.getProperty("user.dir"),"Person/");
        FileSplit filesplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numDatapoints,numLabels, batchSize);
        InputSplit[] inputSplit = filesplit.sample(pathFilter, splitTrainTest, 1-splitTrainTest);

        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);

        log.info("Building model....");


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP).momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(listenerFreq));
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainiter;

        log.info("Training model");
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainiter = new MultipleEpochsIterator(epochs, dataIter,1);
        model.fit(trainiter);

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = model.evaluate(dataIter);
        log.info(eval.stats(true));

        // Example on how to get predict results with trained model
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = model.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");
    }
}
