

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Travel {
	/** The n value */
	private static final int[] cities = {5,10,20,30,40}; // change the number of cities to see the effect
	/**
	 * The test main
	 * @param args ignored
	 */
	public static void main(String[] args) {
		double start, end,trainingTime;
		String results, resultsSummary, finalResults="";
		double averageTime, averageScore, sumTime,sumScore;
		int repeatTime=10;  // each condition will run 10 times and then take average
		//set the parameter for each
		int[] interations= {2,5,10,20,40,80,100,200,500,1000,2000,4000,10000,20000,40000,100000};
		int[] interationsMic= {2,5,10,20,40,80,100,200,500,1000,1500,2000,2500,3000};
		double [] temperature= {0.95,0.65,0.55,0.25,0.01};
		int[] populationSizes={200,100,50};
		double[] geneticRates= { 0.9,0.5,0.1};
		double[] keepRate= { 0.9,0.5,0.1};



		Random random = new Random();
		DecimalFormat dformat = new DecimalFormat("0.000000");
		// create the random points

		for(int N: cities)
		{   System.out.println("----------start City:"+N);
			double[][] points = new double[N][2];
			for (int i = 0; i < points.length; i++) {
				points[i][0] = random.nextDouble();
				points[i][1] = random.nextDouble();   
			}
			// for rhc, sa, and ga we use a permutation based encoding
			TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
			Distribution odd = new DiscretePermutationDistribution(N);
			NeighborFunction nf = new SwapNeighbor();
			MutationFunction mf = new SwapMutation();
			CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
			HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
			GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
			FixedIterationTrainer fit;

      
			for (int i=0;i<interations.length;i++)
			{sumTime=0;
			sumScore=0;
			for(int j=0;j<repeatTime;j++)
			{ 
				RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
				fit = new FixedIterationTrainer(rhc, interations[i]);
				start=System.nanoTime();
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);

				results="cities, "+N +", RHC, "+ "interations,"+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(rhc.getOptimal()))+"\n";
				finalResults=finalResults+results;
				sumTime=sumTime+trainingTime;
				sumScore=sumScore+ef.value(rhc.getOptimal());
			//	System.out.println(results);
				//    System.out.println(ef.value(rhc.getOptimal()));
			}
			resultsSummary="RHCcities, "+N +",Summary, RHCaverage iteration, "+ interations[i]+" , average time, "+ dformat.format (sumTime/repeatTime)+" ,average score is, "+dformat.format(sumScore/repeatTime)+"\n";
			finalResults=finalResults+resultsSummary;
		//	System.out.println(resultsSummary);
			}
			// finish hill climbing   

			for (double t:temperature)
			{
				for (int i=0;i<interations.length;i++)
				{sumTime=0;
				sumScore=0;
				for(int j=0;j<repeatTime;j++)
				{ 
					SimulatedAnnealing sa = new SimulatedAnnealing(1E11, t, hcp);
					fit = new FixedIterationTrainer(sa, interations[i]);
					start=System.nanoTime();
					fit.train();
					end = System.nanoTime();
					trainingTime = end - start;
					trainingTime /= Math.pow(10,9);

					results="cities, "+N +",SA, "+ "temperature,"+t+ "interations"+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(sa.getOptimal()))+"\n";
					finalResults=finalResults+results;
					sumTime=sumTime+trainingTime;
					sumScore=sumScore+ef.value(sa.getOptimal());
					//System.out.println(results);
					//    System.out.println(ef.value(rhc.getOptimal()));
				}
				resultsSummary="SAcities, "+N +",Summary, SA temperature, "+t+ ", iteration, "+ interations[i]+ " ,average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is, " +dformat.format(sumScore/repeatTime)+"\n";
				//System.out.println(resultsSummary);
				finalResults=finalResults+resultsSummary;
				}
			}
			// finish SA


			for (int m=0; m< populationSizes.length;m++)
			{   
				for (int n=0; n< geneticRates.length; n++)
					for (int i=0;i<interationsMic.length;i++)
					{  int mate=(int)(Math.round(populationSizes[m]*geneticRates[n]));
					int mutatation=(int)(Math.round(populationSizes[m]*geneticRates[n]*geneticRates[n]));
					sumTime=0;
					sumScore=0;
					for(int j=0;j<repeatTime;j++)
					{  
						StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSizes[m], mate, mutatation, gap);
						fit = new FixedIterationTrainer(ga, interationsMic[i]);
						start=System.nanoTime();
						fit.train();
						end = System.nanoTime();
						trainingTime = end - start;
						trainingTime /= Math.pow(10,9);
						results="cities, "+N +" ,GA, "+ "poupulation," +populationSizes[m]+" mutation/mate: "+mate + " ,/, " +mutatation +" interations "+interationsMic[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(ga.getOptimal()))+"\n";
						finalResults=finalResults+results;
						sumTime=sumTime+trainingTime;
						sumScore=sumScore+ef.value(ga.getOptimal());
						//System.out.println(results);

						//	System.out.println(ef.value(ga.getOptimal()));
					}
					resultsSummary="GAcities, "+N +",Summary, GA poupulation,"+populationSizes[m]+ ",mate "+ mate + ", interations,"+interationsMic[i]+",average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is, " +dformat.format(sumScore/repeatTime)+"\n";
					//System.out.println(resultsSummary);
					finalResults=finalResults+resultsSummary;
					}
			}


			// for mimic we use a sort encoding
			ef = new TravelingSalesmanSortEvaluationFunction(points);
			int[] ranges = new int[N];
			Arrays.fill(ranges, N);
			odd = new  DiscreteUniformDistribution(ranges);
			Distribution df = new DiscreteDependencyTree(.1, ranges); 
			ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
			//finish other parameter for MIMC start to loop		
			for (int m=0; m< populationSizes.length;m++)
			{   
				for (int n=0; n< keepRate.length; n++)
					for (int i=0;i<interationsMic.length;i++)
					{
						sumTime=0;
						sumScore=0;
						int tokeep=(int)(Math.round(populationSizes[m]*keepRate[n]));
						for(int j=0;j<repeatTime;j++)
						{ 

							MIMIC mimic = new MIMIC(populationSizes[m], tokeep, pop);
							fit = new FixedIterationTrainer(mimic, interationsMic[i]);
							start=System.nanoTime();
							fit.train();
							end = System.nanoTime();
							trainingTime = end - start;
							trainingTime /= Math.pow(10,9);
							results="cities, "+N +", MIMIC, "+ "poupulation," +populationSizes[m]+" tokeep: "+tokeep+ " ,  interations "+interationsMic[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(mimic.getOptimal()))+"\n";
							finalResults=finalResults+results;
							sumTime=sumTime+trainingTime;
							sumScore=sumScore+ef.value(mimic.getOptimal());
							//System.out.println(results);

							//System.out.println(ef.value(mimic.getOptimal()));
						}
						resultsSummary="MIMICcities, "+N +",Summary, mimic poupulation, "+ +populationSizes[m]+" , tokeep,  "+tokeep+ ", interations,"+interationsMic[i]+  ",average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is, " +dformat.format(sumScore/repeatTime)+"\n";
						//System.out.println(resultsSummary );
						finalResults=finalResults+resultsSummary;

					}
			}	
			write_output_to_file(finalResults,"travler.csv",true);
		}

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
