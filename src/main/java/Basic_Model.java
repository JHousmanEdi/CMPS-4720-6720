import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by jason on 3/12/17.
 */
public class Basic_Model {
    private static final Logger log = LoggerFactory.getLogger(Basic_Model.class);

    public static void main(String[] args) throws Exception {
        Nd4j.create(1);

        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true)
                .setMaximumDeviceCache(2L*1024L*1024L*1024L)
                .allowCrossDeviceAccess(true);


        log.info("Load data....");

        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("Image_Data_clean.csv").getFile()));
        //reader,label index,number of possible labels

        int labelIndex= 29;
        int numClasses = 276;
        int batchSize=99535;


        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        //get the dataset using the record reader. The datasetiterator handles vectorization
        DataSet allData = iterator.next();
        // Customizing params

        allData.shuffle(); //Randommly shuffles Data
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); //Splits data into test and trainset

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testdata = testAndTrain.getTest();
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);
        normalizer.transform(trainingData);
        normalizer.transform(testdata);

        final int numInputs = 29;
        int outputNum = 276;
        int iterations = 1000;
        long seed = 6;
        int nEpochs = 5;

        log.info("Build Model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                    .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(outputNum).build())
                .backprop(true).pretrain(false).build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(100));
        long timeX = System.currentTimeMillis();

        for (int i = 0; i < 1; i++){
            long time1 = System.currentTimeMillis();

            model.fit(trainingData);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed Epoch {}, time: {} ***",i, (time2-time1));
        }
        long timeY = System.currentTimeMillis();
        log.info("***Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        INDArray output = model.output(testdata.getFeatures());
        Evaluation eval = new Evaluation(outputNum);
        eval.eval(testdata.getLabels(), output);
        log.info(eval.stats());


        }
    }