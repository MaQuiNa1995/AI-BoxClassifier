package maquina1995.deep.learning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * En esta clase se eliminan los comentarios porque ya está explicado en {@link MainColor#main(String...)}
 * 
 * Este ejemplo clasifica cajas por:
 * - Color
 * - Tamaño
 */
public class MainColorTamano {

	public static void main(String[] args) {
		// --- 1. Creacion del Modelo de Entrenamiento ---
		DataSetIterator iterator = createModel();

		// --- 2. Configuración de la Red Neuronal ---
		MultiLayerNetwork model = configureNeuralNet();

		// --- 3. Uso del Modelo de Entrenamiento Para Entrenar ---
		trainingModel(iterator, model);

		// --- Uso Del Modelo Con Datos Reales ---
		useModelForPredictions(model);
	}

	private static DataSetIterator createModel() {

		System.out.println("Se empieza a crear el modelo");

		List<DataSet> data = new ArrayList<>();
        List<String> colores = List.of("Rojo", "Verde", "Azul");
        List<String> tamanos = List.of("Grande", "Mediano", "Pequeño");

        System.out.println("--- Modelo ---");
        
        AtomicInteger colorIndex = new AtomicInteger(0);
        AtomicInteger sizeindex = new AtomicInteger(0);
        
        colores.forEach(color -> 
        	tamanos.forEach(tamano -> {
        		
                double[] input = new double[6];
                
	            switch (color) {
	                case "Rojo" -> {
	                	input[0] = 1;
	                	colorIndex.set(0);
	                }
	                case "Verde" -> {
	                	input[1] = 1;
	                	colorIndex.set(3);
	                }
	                case "Azul" -> {
	                	input[2] = 1;
	                	colorIndex.set(6);
	                }
	            }
	
	            switch (tamano) {
	                case "Grande" -> {
	                	input[3] = 1;
	                	sizeindex.set(0);
	                }
	                case "Mediano" -> {
	                	input[4] = 1;
	                	sizeindex.set(1);
	                }
	                case "Pequeño" -> {
	                	input[5] = 1;
	                	sizeindex.set(2);
	                }
	            }
	            
	            System.out.print(Arrays.toString(input) + " " + color + ","+ tamano);

                INDArray inputNdArray = Nd4j.create(input);

                double[] output = new double[9];
                
                int indexClassifier = colorIndex.get() + sizeindex.get();
                output[indexClassifier] = 1;

                System.out.println(" -> Solución: " + Arrays.toString(output));
                
                INDArray outputNdArray = Nd4j.create(output);
                data.add(new DataSet(inputNdArray, outputNdArray));
        	})
        );
        
        System.out.println("---------------------");
        
		DataSetIterator iterator = new ListDataSetIterator<>(data, 9);
		
		System.out.println("Se acabó de crear el modelo");
		
		return iterator;
	}

	private static MultiLayerNetwork configureNeuralNet() {
		
		System.out.println("Se empieza a crear la red neuronal");
		
		int numInputs = 6;
		int numOutputs = 9;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam())
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numInputs)
						.nOut(10)
						.activation(Activation.RELU)
						.build())
				.layer(1, new DenseLayer.Builder()
						.nIn(10)
						.nOut(9)
						.activation(Activation.RELU)
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(9)
						.nOut(numOutputs)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));
		
		System.out.println("Se creó la red neuronal");
		
		return model;
	}
	
	private static void useModelForPredictions(MultiLayerNetwork model) {

		List<String> colors = List.of("Rojo y Grande", "Rojo y Mediano", "Rojo y Pequeño", "Verde y Grande",
				"Verde y Mediano", "Verde y Pequeño", "Azul y Grande", "Azul y Mediano", "Azul y Pequeño");

		// Objeto Verde, Pequeño
		double[] cajaAnalizar = new double[] { 0, 1, 0, 0, 1, 0 };

		INDArray inputPrueba = Nd4j.create(cajaAnalizar);
		INDArray prediction = model.output(inputPrueba);

		System.out.println("Predicción:");
		AtomicInteger counter = new AtomicInteger(0);
		colors.stream()
				.forEach(color -> System.out.println("Color: " + color + " -> "
						+ String.format("%.2f", prediction.getFloat(counter.getAndIncrement()) * 100) + "%"));

		int predictedClass = Nd4j.argMax(prediction, 1)
				.getInt(0);

		System.out.println("Con los datos de la caja: " + Arrays.toString(cajaAnalizar) + " la clasificamos como: "
				+ colors.get(predictedClass));
	}

	private static void trainingModel(DataSetIterator iterator, MultiLayerNetwork model) {
		System.out.println("Se empieza a entrenar el modelo");
		
		int numEpochs = 100;
		for (int i = 0; i < numEpochs; i++) {
			iterator.reset();
			model.fit(iterator);
		}
		
		System.out.println("Ya se ha acabado de entrenar el modelo");
	}

}
