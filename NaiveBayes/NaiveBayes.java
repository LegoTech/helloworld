import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class NaiveBayes {
	//global variables
	int[] docLabelArray;
	@SuppressWarnings("unchecked")
	HashMap<Integer,Integer>[] n_k = new HashMap[21];//n_k[i].get(k): number of word w_k in each class i
	double[] priorData = new double[20];
	int[] n_wordClass=new int[21];//n_wordClass[i]: number of times any words occur in all documents in class i
	int wordCount;

	public NaiveBayes(){
		this.wordCount = 0;
		
		for(int i = 0;i<21;++i){
			n_k[i]=new HashMap<Integer,Integer>();
		}

		//calculate words count
		File vocabulary = new File("vocabulary.txt");
		try{
		BufferedReader br=new BufferedReader(new FileReader(vocabulary));
		while(br.readLine()!=null){
			++this.wordCount;
		}
		br.close();
		}catch(IOException e){
			e.printStackTrace();
		}	
//		System.out.println("size of vocabulary = "+wordCount);
	}
	
	private void printPrior(){
		List<Integer> docLabel = new ArrayList<Integer>();
		int docCount=0;
		int[] w_i=new int[21]; 									// w_i[i]: # of documents of class i
		File train_label = new File("train_label.csv");
		String temp = null;
		int index;
		
		try{
			BufferedReader br=new BufferedReader(new FileReader(train_label));
		
			while((temp=br.readLine())!=null){
				index = Integer.parseInt(temp);
				docLabel.add(index);
				++w_i[index];
				++docCount;
			}
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}
		
		System.out.println("Class priors:");
		for(int i=1; i<21; i++){//
			priorData[i-1] = (double)w_i[i]/docCount;
			System.out.println("P(Omega = "+ i +") = " +priorData[i-1]);
		}
		docLabelArray = docLabel.stream().mapToInt(i->i).toArray();		//ArrayList to Array
	}	//end of printprior
	
	private void getCounts(){
		File train_data = new File("train_data.csv");
		
		String temp;
		int docId;
		int docLabel;//find document class in docLabelArray, based on doc ID
		int wID;//word ID
		int nk;//number of times word wk occurs in all documents in class
		try{
			BufferedReader br=new BufferedReader(new FileReader(train_data));

			while((temp=br.readLine())!=null){				
				String[] elements = temp.split(",");
				docId = Integer.parseInt(elements[0])-1;
				docLabel = docLabelArray[docId];						//find the class of the doc in the docLabelArray
				wID = Integer.parseInt(elements[1]);
				nk = Integer.parseInt(elements[2]);
				n_wordClass[docLabel]+=nk;
				
				if(n_k[docLabel].get(wID)!=null){
					n_k[docLabel].put(wID, n_k[docLabel].get(wID)+nk);		//n_k[i].get(k): number of word w_k in each class i
				}else{
					n_k[docLabel].put(wID, nk);
				}
			}	
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}

	}	//end of getCounts

	private double getLnPmle(int wID, int ClassId){
		if(n_k[ClassId].get(wID)!=null){
			return Math.log((double)n_k[ClassId].get(wID)/n_wordClass[ClassId]);
		}else{
			return -Double.MAX_VALUE;
		}
	}

	private double getLnPbe(int wID, int ClassId){
		if(n_k[ClassId].get(wID)!=null){
			return Math.log(((double)n_k[ClassId].get(wID)+1)/(wordCount+n_wordClass[ClassId]));
		}else{
			return Math.log(((double)1)/(wordCount+n_wordClass[ClassId]));
		}

	}
	
	private void classifier(String DataFileName, String LabelFileName, String estimatorName){
		File train_data = new File(""+DataFileName);
		List<int[]> data = new ArrayList<int[]>();
		List<double[]> wNBs = new ArrayList<double[]>();	//wNBs[i]: doc i prosterior probability
		List<Integer> classificated = new ArrayList<Integer>();	//label results
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(16);
		
		String temp = null;
		try{
		BufferedReader br=new BufferedReader(new FileReader(train_data));
		while((temp=br.readLine())!=null){
			String[] elements = temp.split(",");
			data.add(new int[]{Integer.parseInt(elements[0]),Integer.parseInt(elements[1]),Integer.parseInt(elements[2])});//doc id, word id, number of this word in this doc
		}
		br.close();
		}catch(IOException e){
			e.printStackTrace();
		}
//		System.out.println(data.size());	//dataSize

		//classifier
		for(int i=0, base=0, datasize = data.size(); i<datasize; i++){
			if(data.get(base)[0]!=data.get(i)[0]||i==datasize-1){
				double[] startp = priorData.clone();
				wNBs.add(startp);//20 class for each news
				classificated.add(0);			//stores classification result
				final int threadbase = base;
				final int threadbound = i;
				final int wNBsIndex = wNBs.size()-1;
				fixedThreadPool.execute(new Runnable() {					
					@Override
					public void run() {
							double tempmaxPbe=-Double.MAX_VALUE;
//						try {
							for(int j =1; j<21; ++j){
								wNBs.get(wNBsIndex)[j-1]=Math.log(wNBs.get(wNBsIndex)[j-1]);//column initialization
								for(int k = threadbase; k<threadbound; ++k){
									if (estimatorName=="mle"){
										wNBs.get(wNBsIndex)[j-1]+=(getLnPmle(data.get(k)[1], j)*data.get(k)[2]);
									}else if(estimatorName=="be"){
										wNBs.get(wNBsIndex)[j-1]+=(getLnPbe(data.get(k)[1], j)*data.get(k)[2]);
									}
								}//end of inner for-loop

								//argmax
								if(tempmaxPbe<wNBs.get(wNBsIndex)[j-1]){
									tempmaxPbe=wNBs.get(wNBsIndex)[j-1];
									classificated.set(wNBsIndex,j-1);
								}
							}
//						}catch(Exception e){
//							e.printStackTrace();
//						}
					}
				});	
				base =i;				
			}
		}
		fixedThreadPool.shutdown();
		
		try{
			fixedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
		}
		catch (InterruptedException e){
			e.printStackTrace();
		}
		
		int[][] c_matrix = new int[20][20];
		
		List<Integer> docLabel = new ArrayList<Integer>();

		File label = new File(LabelFileName);
		
		
		int index;
		int docCount=0;
		try{
			BufferedReader br=new BufferedReader(new FileReader(label));
		
			while((temp=br.readLine())!=null){
				index = Integer.parseInt(temp);
				docLabel.add(index);
				++docCount;
			}
			br.close();
		}catch(IOException e){
			e.printStackTrace();
		}

//		System.out.println(docCount);	//docSize
		
		docLabelArray = docLabel.stream().mapToInt(i->i).toArray();

		for(int i = 0,bound = classificated.size(); i<bound;++i){
			++c_matrix[docLabelArray[i]-1][classificated.get(i)];
		}
		int correct_c=0;
		for(int i =0; i<20;++i){
			correct_c+=c_matrix[i][i];
		}
		System.out.println(DataFileName + " on "+ estimatorName +" estimator");
		System.out.println("Overall Accuracy : "+ (double)correct_c/docCount);
		for(int i =0; i < 20; i++){
			int sum=0;
			for(int j=0;j<20; j++){
				sum+=c_matrix[i][j];
			}
			System.out.println("Group " +i+" : "+(double)c_matrix[i][i]/sum);
		}
		
		System.out.println("Confusion matrix");
		for(int i =0; i< 20; i++){
			for(int j =0; j<20; j++){
				System.out.print(c_matrix[i][j]+"\t");
			}
			System.out.println("");
		}
	}

	public static void main(String args[]){
		NaiveBayes Naive= new NaiveBayes();
		Naive.printPrior();
		Naive.getCounts();

		//compare Pmle with Pbe
//		for(int i =1; i<2; ++i){
////			for(int key=0; key<wordCount; key++){
//			for(Integer key:n_k[i].keySet()){
//				System.out.println(n_k[i].get(key));
//				System.out.println("P_MLE("+key+"|Omega"+i+")"+getLnPmle(key, i));
//				System.out.println("P_BE("+key+"|Omega"+i+")"+getLnPbe(key, i));
//			}
//		}

		Naive.classifier("train_data.csv","train_label.csv", "be");
		Naive.classifier("test_data.csv","test_label.csv", "be");

		Naive.classifier("train_data.csv","train_label.csv", "mle");
		Naive.classifier("test_data.csv","test_label.csv", "mle");

	}
}