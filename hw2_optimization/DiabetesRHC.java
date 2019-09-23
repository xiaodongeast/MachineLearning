

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class DiabetesRHC {
	private static int rows=1151;
	private static int attributesNumber=19;
	// private static Instance[] instances = initializeInstances();

	private static double[][][] values=readData();
	private static double[][][] shuffled=shuffleAttribute(values);

	private static Instance[] instances = initializeInstances(shuffled);

	private static Instance[] trainData=Arrays.copyOfRange(instances, 0, 805);
	private static Instance[] test = Arrays.copyOfRange(instances, 806, 1151);

	private static int inputLayer = 19, hiddenLayer1 = 50, hiddenLayer2=50, outputLayer = 1, trainingIterations = 1000;
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

	private static ErrorMeasure measure = new SumOfSquaresError();

	//this is the only place to set the trainData
	private static DataSet set = new DataSet(trainData);

	private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
	private static String[] oaNames = {"SA", "RHC", "GA"};
	//private static String results = "";

	private static DecimalFormat df = new DecimalFormat("0.000");

	public static void main(String[] args) {
		int repeats=10;

		int[] inter = {2,5,10,20,40,80,100,200,500,1000,1500,2000,2500,3000};
	//	int[] inter = {2,5,10,40,100,500,1000};
		

		String finalResult="";
		double trainingTime, testingTime, correct = 0, incorrect = 0;
		double[] testResult=new double[3];
		for(int i = 0; i < oa.length; i++) {
			networks[i] = factory.createClassificationNetwork(
					new int[] {inputLayer, hiddenLayer1, hiddenLayer2, outputLayer});
			nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
		}
		oa[0] = new SimulatedAnnealing(1E11, 0.95, nnop[1]);
		oa[1] = new RandomizedHillClimbing(nnop[0]);





		for (int j=0;  j< inter.length;j++)
		{ 	
			System.out.println("################# iterations  " +inter[j]  );
			double finalTrainTime=0, finalTrainScore=0, finalTestTime=0, finalTestScore=0;
		 
					for (int repeat=0; repeat<repeats; repeat++)
					{ 
						System.out.println("---------------round " +repeat);
						trainingTime= buildModel( oa[1], "GA", networks[1],inter[j]);
						testResult= testModel(trainData,networks[1]  );
						finalTrainTime=finalTrainTime+trainingTime;
						finalTrainScore=finalTrainScore+(testResult[0]/(testResult[0]+testResult[1])*100);
						String results;
						results=  "Training,"+ "RHC"+ ",iterateion,"+ inter[j]+  ",Correctly classified, " + testResult[0] +
								",Incorrectly classified, " + testResult[1] + " ,Percent correctly classified:, "
								+ df.format(testResult[0]/(testResult[0]+testResult[1])*100) + ",%Training time:, " + df.format(trainingTime)
								+ ",Testing time:, " + df.format(testResult[2])+"\n";
						//	System.out.println(results);


						testResult= testModel(test,networks[1]  );

						results=  "Predicting,"+ "RHC," +",iterateion,"+ inter[j]+  ",Correctly classified, " + testResult[0] +
								",Incorrectly classified, " + testResult[1] + " ,Percent correctly classified:, "
								+ df.format(testResult[0]/(testResult[0]+testResult[1])*100) + ",%Training time:, " + df.format(trainingTime)
								+ ",Testing time:, " + df.format(testResult[2])+"\n";
						//	System.out.println(results);
						finalTestTime=finalTestTime+testResult[2];
						finalTestScore=finalTestScore+(testResult[0]/(testResult[0]+testResult[1])*100);

						//	System.out.println(results);
					}

					finalTrainScore=finalTrainScore/repeats;
					finalTestScore=finalTestScore/repeats;
					finalTestTime=finalTestTime/repeats;
					finalTrainTime=finalTrainTime/repeats;
					finalResult=finalResult+"RHC,"+ ",iteration," +inter[j] +",  average trainScore,"+finalTrainScore+ ",averageTestScore,"+finalTestScore+
							",finalTrainTime, " +finalTrainTime+",finalTesttime,"+finalTestTime +"\n";
					// System.out.println(finalResult);
				}

			
		
		write_output_to_file(finalResult,"diabetesRHCResult.csv",true);

	}
	//no global
	private static  double buildModel(OptimizationAlgorithm oa, String oaNames, BackPropagationNetwork networks, int interations) {

		double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
		train(oa, networks, oaNames, interations); //trainer.train();
		end = System.nanoTime();
		trainingTime = end - start;
		trainingTime /= Math.pow(10,9);

		Instance optimalInstance = oa.getOptimal();
		networks.setWeights(optimalInstance.getData());
		return trainingTime;
	}
	// no global
	private static double[]  testModel(Instance[] sample,BackPropagationNetwork networks  ) {

		double predicted, actual,start = System.nanoTime(), end, testingTime,correct=0,incorrect=0;
		double[] testResult=new double[3];
		for(int j = 0; j < sample.length; j++) {
			networks.setInputValues(sample[j].getData());
			networks.run();

			actual = Double.parseDouble(sample[j].getLabel().toString());
			predicted = Double.parseDouble(networks.getOutputValues().toString());

			double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

		}
		end = System.nanoTime();
		testingTime = end - start;
		testingTime /= Math.pow(10,9);
		testResult[0]=correct;
		testResult[1]=incorrect;
		testResult[2]=testingTime;

		return testResult;

	}
	// no global
	private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int interations) {
		//		System.out.println("\nError results for " + oaName + "\n---------------------------");

		for(int i = 0; i < interations; i++) {
			oa.train();
			/*
            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
			 */
		}
	}

	//global
	private static double[][][] readData ()
	{   double[][][] attributes = new double[rows][][];
	String show;

	try {
		BufferedReader br = new BufferedReader(new FileReader(new File("src/dia.txt")));

		for(int i = 0; i < attributes.length; i++) {
			Scanner scan = new Scanner(br.readLine());
			scan.useDelimiter(",");

			attributes[i] = new double[2][];
			attributes[i][0] = new double[attributesNumber]; // 7 attributes
			attributes[i][1] = new double[1];

			for(int j = 0; j < attributesNumber; j++)
				attributes[i][0][j] = Double.parseDouble(scan.next());

			attributes[i][1][0] = Double.parseDouble(scan.next());
		}
	}
	catch(Exception e) {
		e.printStackTrace();
	}

	return attributes;
	}


	private static Instance[] initializeInstances(double[][][] attributes) {
		Instance[] instances = new Instance[attributes.length];
		for(int i = 0; i < instances.length; i++) {
			instances[i] = new Instance(attributes[i][0]);
			instances[i].setLabel(new Instance(attributes[i][1][0]==0 ? 0 : 1));
		}

		return instances;
	}


	private static double[][][] shuffleAttribute(double[][][] arrayOrig)
	{
		double[][][] array=Arrays.copyOf(arrayOrig, rows);
		Random random = new Random();
		int index;
		double[][] temp = new double[2][];
		int count = array.length-1;
		for (int i = count; i > 0; i--) {
			index=random.nextInt(i+1);
			temp=array[i];
			array[i]=array[index];
			array[index]=temp;

		}

		return array;

	}



	public static void write_output_to_file( String result,String fileName, Boolean final_result) {

		try {
			if (final_result) {     
				PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(fileName)));
				synchronized (pwtr) {
					pwtr.println( result);
					pwtr.close();
				}
			}  
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}

