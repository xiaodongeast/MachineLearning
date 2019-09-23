 

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Knapsack {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int[] totalItems = {2,5,10,20,30,40,50};
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 40;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 40;
    /** The maximum weight for the knapsack */
  

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
		double [] cooling= {0.9,0.5,0.2};
		int[] populationSizes={200,100,50};
		double[] geneticRates= { 0.8,0.4,0.2};
		double[] keepRate= { 0.8,0.4,0.2};
		double MAX_KNAPSACK_WEIGHT;
		DecimalFormat dformat = new DecimalFormat("0.000000");
    	
		// start set the bag limitation and the items.
		for(int NUM_ITEMS: totalItems)
			{ MAX_KNAPSACK_WEIGHT = MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
 
        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        FixedIterationTrainer fit;
        System.out.println("----------------start hill climbing for items: "+NUM_ITEMS);
        
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
       // System.out.println(ef.value(rhc.getOptimal()));
		results="items, "+NUM_ITEMS +", RHC, "+ "interations,"+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(rhc.getOptimal()))+"\n";
		//finalResults=finalResults+results;
		sumTime=sumTime+trainingTime;
		sumScore=sumScore+ef.value(rhc.getOptimal());
	//	System.out.println(results);
	}
	resultsSummary="RHCcities, "+NUM_ITEMS +",Summary, RHCaverage iteration, "+ interations[i]+" , average time, "+ dformat.format (sumTime/repeatTime)+" ,average score is, "+dformat.format(sumScore/repeatTime)+"\n";
	finalResults=finalResults+resultsSummary;
	System.out.println(resultsSummary);
	}
	// 
        
        
        
        System.out.println("----------------start SA climbing for items: " +NUM_ITEMS);
        
    	for (double t:cooling)
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

		results="cities, "+NUM_ITEMS +",SA, "+ "temperature,"+t+ "interations"+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(sa.getOptimal()))+"\n";
	//	finalResults=finalResults+results;
		sumTime=sumTime+trainingTime;
		sumScore=sumScore+ef.value(sa.getOptimal());
		//System.out.println(ef.value(sa.getOptimal()));
			}
			
			resultsSummary="SAcities, "+NUM_ITEMS+",Summary, SA Cooling, "+t+ ", iteration, "+ interations[i]+ " ,average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is " +dformat.format(sumScore/repeatTime)+"\n";
			System.out.println(resultsSummary);
			finalResults=finalResults+resultsSummary;
			}
		}
		// finish SA
        
    	 System.out.println("----------------start GA for items:  " +NUM_ITEMS);
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
        //System.out.println("start genetic problem----------------")
					StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSizes[m], mate, mutatation, gap);
					fit = new FixedIterationTrainer(ga, interationsMic[i]);
					start=System.nanoTime();
					fit.train();
					end = System.nanoTime();
					trainingTime = end - start;
					trainingTime /= Math.pow(10,9);
					results="cities, "+NUM_ITEMS +" ,GA, "+ "poupulation," +populationSizes[m]+" mutation/mate: "+mate + " ,/, " +mutatation +" interations "+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(ga.getOptimal()))+"\n";
					//finalResults=finalResults+results;
					sumTime=sumTime+trainingTime;
					sumScore=sumScore+ef.value(ga.getOptimal());
				//	System.out.println(ef.value(ga.getOptimal()));
				}
       		resultsSummary="GAcities, "+NUM_ITEMS +",Summary, GA poupulation,"+populationSizes[m]+ ",mate "+ mate + ", interations,"+interationsMic[i]+",average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is, " +dformat.format(sumScore/repeatTime)+"\n";
					System.out.println(resultsSummary);
					finalResults=finalResults+resultsSummary;
					}
			}
        
		
		 System.out.println("----------------start MIMIC climbing for: "+NUM_ITEMS);
        
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
							results="cities, "+NUM_ITEMS +", MIMIC, "+ "poupulation," +populationSizes[m]+" tokeep: "+tokeep+ " ,  interations "+interations[i]+" , repeat "+ j+" , " +dformat.format(trainingTime) +" ," + dformat.format(ef.value(mimic.getOptimal()))+"\n";
						//	finalResults=finalResults+results;
							sumTime=sumTime+trainingTime;
							sumScore=sumScore+ef.value(mimic.getOptimal());
							//System.out.println(results);

						//	System.out.println(ef.value(mimic.getOptimal()));
						}
						resultsSummary="MIMICcities, "+NUM_ITEMS +",Summary, mimic poupulation, "+ +populationSizes[m]+" , tokeep,  "+tokeep+ ", interations,"+interationsMic[i]+  ",average time is, "+ dformat.format (sumTime/repeatTime)+ ",average score is, " +dformat.format(sumScore/repeatTime)+"\n";
				  	System.out.println(resultsSummary );
						finalResults=finalResults+resultsSummary;

					}
			}	
			write_output_to_file(finalResults,"knapsack.csv",true);
		
			}
			}
    // output method
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
