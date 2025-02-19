package maquina1995.deep.learning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

public class Main {

	public static void main(String[] args) {

		// --- 1. Creacion del Modelo de Entrenamiento ---
		DataSetIterator iterator = createModel();

		// --- 2. Configuración de la Red Neuronal ---
		MultiLayerNetwork model = configureNeuralNet();

		// --- 3. Uso del Modelo de Entrenamiento Para Entrenar ---
		trainingModel(iterator, model);

		// --- Uso Del Modelo Con Datos Reales ---
		useModelForPredictions(model);

		/**
		 * Hay que tener en cuenta que contra mas datos tenga el modelo mejores
		 * predicciones hará
		 * 
		 * En este caso para que pueda predecir cajas mas variadas contra mas datos
		 * tengamos de cajas mejor de hecho lo idea sería tener 1 ejemplo de cada tamaño
		 * y color
		 * 
		 * Es decir tener 9 -> 3 (tamaños) * 3 (colores) = 9 posibilidades
		 */

	}

	private static DataSetIterator createModel() {
		// Creamos la lista del modelo de entrenamiento
		List<DataSet> data = new ArrayList<>();

		/**
		 * Usamos 1 y 0 en un array unidimensional para representar algun objeto en este
		 * caso cajas teniendo en cuenta estas pautas: Cada caja tiene: - 1 Color - 1
		 * Tamaño
		 * 
		 * Para representar esto en un array unidimensional vamos a seguir: - Color:
		 * Modelo RGB (Red, Green, Blue) - Tamaño: Grande, Mediano, Pequeño
		 * 
		 * De tal manera que si lo juntamos en nuestro array quedaría asi: esRojo,
		 * esVerde, esAzul, esGrande, esMediano, esPequeño
		 * 
		 * Y pasado a como lo interpretaría el programa para una caja: - Color Verde -
		 * Tamaño Grande
		 * 
		 * 0, 1, 0, 1, 0, 0
		 * 
		 * Para que la librería interprete estos datos es necesario entrenarla
		 * proporcionandola un ejemplo y una solución como hicimos arriba
		 * 
		 * le proporcionas 2 arrays: - Ejemplo: new double[] { 1, 0, 0, 1, 0, 0 } ->
		 * CajaEjemplo - Solución: new double[] { 1, 0, 0 } -> Rojo, Grande
		 */

		// Ejemplo caja 1: Rojo, Grande -> [1, 0, 0, 1, 0, 0]
		INDArray input1 = Nd4j.create(new double[] { 1, 0, 0, 1, 0, 0 });
		// Solución esRoja
		INDArray output1 = Nd4j.create(new double[] { 1, 0, 0 });
		// Añadimos el ejemplo y la solucion a la lista
		data.add(new DataSet(input1, output1));

		// Ejemplo caja 2: Verde, Mediana -> [1, 0, 0, 0, 1, 0]
		INDArray input2 = Nd4j.create(new double[] { 0, 1, 0, 0, 1, 0 });
		// Solución esVerde
		INDArray output2 = Nd4j.create(new double[] { 0, 1, 0 });
		// Añadimos el ejemplo y la solucion a la lista
		data.add(new DataSet(input2, output2));

		// Ejemplo caja 3: Azul, Pequeña -> [0, 0, 1, 0, 0, 1]
		INDArray input3 = Nd4j.create(new double[] { 0, 0, 1, 0, 0, 1 });
		// Solución esAzul
		INDArray output3 = Nd4j.create(new double[] { 0, 0, 1 });
		// Añadimos el ejemplo y la solucion a la lista
		data.add(new DataSet(input3, output3));

		/**
		 * Ahora le decimos a la librería como queremos que procese los datos de
		 * entrenamiento El numero que le ponemos indica la cantidad de elementos a
		 * procesar a la vez
		 */
		DataSetIterator iterator = new ListDataSetIterator<>(data, 1);
		return iterator;
	}

	private static void useModelForPredictions(MultiLayerNetwork model) {
		// Objeto Rojo, Grande
		double[] cajaAnalizar = new double[] { 1, 0, 0, 1, 0, 0 };

		INDArray inputPrueba = Nd4j.create(cajaAnalizar);
		// Le pasamos la caja y nos da la predicción
		INDArray prediction = model.output(inputPrueba);

		// Imprimimos por pantalla la predicción para cada color para esta caja
		// obtendríamos algo parecido a esto: [1.00, 0.00, 0.00]
		System.out.println("Predicción: " + prediction);

		// Para obtener en este caso el color usaremos la siguiente línea para obtener
		// el valor mas alto mas cercano a 1
		int predictedClass = Nd4j.argMax(prediction, 1)
				.getInt(0);

		// Ahora para sacar de manera visual por pantalla el color de la caja creamos un
		// array de los colores en orden RGB (Vamos el mismo orden que definimos arriba)
		List<String> colors = List.of("Rojo", "Verde", "Azul");

		// Ahora con un simple get obtendremos por pantalla el color de esa caja
		System.out.println("Con los datos de la caja: " + Arrays.asList(cajaAnalizar) + " la clasificamos como: "
				+ colors.get(predictedClass));
	}

	private static MultiLayerNetwork configureNeuralNet() {
		// Esta variable representa la cantidad de valores que tendra el array que
		// representara la caja en este caso entran 6 porque: 3 colores + 3 tamaños
		int numInputs = 6;
		// Y esta variable representa la cantidad de tipos de cajas que queremos
		// clasificar
		// Ahora solo nos preocupamos de los colores de las cajas por lo tanto 3
		int numOutputs = 3;

		// Este objeto representa la configuracion de la red neuronal
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				// El seed en si no es necesario es la semilla por la cual se irán creando los
				// numeros aleatorios funciona de igual manera que en la clase java.util.Random
				// sirve para que genere los mismos numeros aleatorios en distintas ejecuciones
				// se puede usar con finalidad de testing si todas las ejecuciones da los mismos
				// numeros es mas facil de testear
				.seed(123)
				/**
				 * Aqui configuramos los pesos iniciales de la red neuronal en este caso Xavier
				 * 
				 * Imagina una red neuronal como un grupo de personas (neuronas) que trabajan
				 * juntas para resolver un problema
				 * 
				 * Cada persona tiene una cierta influencia en la decisión final, algunas más
				 * que otras
				 * 
				 * Los "pesos" en una red neuronal son como la influencia que tiene cada persona
				 * en la decisión final.
				 * 
				 * Ahora Xavier lo que hace es que todas las neuronas primarias (las primeras)
				 * aprendan y no solo las que están mas cerca de la solución final (mas o menos
				 * no se explicarlo mejor)
				 */
				.weightInit(WeightInit.XAVIER)
				/**
				 * Aqui configuramos el algoritmo de optimizacion de la red neuronal lo que
				 * intenta Adam (Algoritmo) es regular el aprendizaje y repartirlo no solo a las
				 * primeras neuronas ni a las del final sino a todas por igual reduciendo la
				 * tasa de error
				 */
				.updater(new Adam())
				// Aqui le decimos a la red neuronal que es multicapa osea que tiene mas de 1
				// layer
				.list()
				// Aqui construimos cada capa:

				/**
				 * Capa 0:
				 * 
				 * .nIn(numInputs) -> numero de posibilidades de caracteristicas de la caja en
				 * este caso 6 .nOut(10) -> numero de neuronas que tendrá esta capa de salida
				 * .activation(Activation.RELU) ->
				 * 
				 * (Rectified Linear Unit) es una función que introduce no linealidad en la red
				 * es decir que los datos no se pasan a la siguiente si capa si no es la
				 * adecuada
				 * 
				 * Nos referimos con esto a que si esa neurona se encarga de las cajas rojas y
				 * la llega una que piensa que sea otro color devuelve 0
				 * 
				 * La función ReLU devuelve 0 si la entrada es negativa, y la entrada misma si
				 * es positiva.
				 * 
				 * Hay una foto en el readme que explica de manera visual la red neuronal el
				 * porque de:
				 * 
				 * Capa 0: in -> 6 (ya lo explicamos antes)
				 * 
				 * Capa 0: out -> 10 ( que sea 10 es cuestion de diseño personal y de ensayo y
				 * pruebas para ver que valores funcionan mejor)
				 * 
				 * Capa 1: in -> 10 Este tiene que ser el mismo que la salida de la anterior
				 * capa
				 * 
				 * Capa 1: out -> 7 ( que sea 7 es cuestion de diseño personal y de ensayo y
				 * pruebas para ver que valores funcionan mejor)
				 */
				.layer(0, new DenseLayer.Builder().nIn(numInputs)
						.nOut(10)
						.activation(Activation.RELU)
						.build())
				// Capa 1
				.layer(1, new DenseLayer.Builder().nIn(10)
						.nOut(7)
						.activation(Activation.RELU)
						.build())
				/**
				 * Capa 2 final
				 * 
				 * Se pueden tener mas y mas capas pero en este caso tenemos 3
				 * 
				 * LossFunctions.LossFunction.MCXENT -> "Multi-Class Cross Entropy" (Entropía
				 * Cruzada Multiclase). Es una función de pérdida que se utiliza comúnmente en
				 * problemas de clasificación multiclase
				 * 
				 * .nIn(7) -> Este tiene que ser el mismo que la salida de la anterior capa
				 * 
				 * .nOut(numOutputs) -> Este tiene que ser el numero de características por las
				 * que queremos clasificar las cajas en este caso (Rojo, Verde, Azul)
				 * 
				 * .activation(Activation.SOFTMAX) es el algoritmo usado para problemas de
				 * clasificacion multiclase es decir el adecuado para cuando quieres clasificar
				 * un conjunto de datos en una serie pre-definida de categorías
				 */
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(7)
						.nOut(numOutputs)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		// Creamos la red neuronal con la configuracion previa
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		// Lo inicializamos
		model.init();

		/**
		 * Añadimos listener que imprime el "score" (la función de pérdida) del modelo
		 * cada cierto número de iteraciones en este caso 10
		 * 
		 * Permite monitorear el progreso del entrenamiento al ver cómo el score
		 * disminuye con el tiempo, puedes tener una idea de si el modelo está
		 * aprendiendo correctamente
		 */
		model.setListeners(new ScoreIterationListener(10));
		return model;
	}

	private static void trainingModel(DataSetIterator iterator, MultiLayerNetwork model) {
		/**
		 * Este numero representa las epocas de la red neuronal
		 * 
		 * Con épocas nos referimos a un ciclo completo de entrenamiento que pasa por
		 * todos los datos de entrenamiento una vez
		 * 
		 * En cada época, el modelo ve todos los ejemplos de entrenamiento, ajusta sus
		 * pesos y aprende de los datos
		 */
		int numEpochs = 100;

		for (int i = 0; i < numEpochs; i++) {
			// Como estamos usando un bucle que consume el iterador de la lista cada
			// "pasada" lo reiniciamos para que esté listo para la siguiente iteración
			iterator.reset();
			// Aqui realizamos 1 epoca de entrenamiento para entrenar el modelo
			model.fit(iterator);
		}
	}

}
