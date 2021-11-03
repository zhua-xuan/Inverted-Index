import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URI;
import java.util.*;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.util.StringUtils;


public class InvertedIndex {

    public static class InvertedIndexMapper
            extends Mapper<Object, Text, Text, IntWritable>{

        static enum CountersEnum { INPUT_WORDS }

        private final static IntWritable one = new IntWritable(1);
        private Text keyInfo = new Text();
        private Text valueInfo = new Text();
        private Set<String> patternsToSkip = new HashSet<String>();
        private Set<String> punctuations = new HashSet<String>();
        private Configuration conf;
        private BufferedReader fis;
        private FileSplit split;
        @Override
        public void setup(Context context) throws IOException,
                InterruptedException {
            conf = context.getConfiguration();

            if (conf.getBoolean("wordcount.skip.patterns", false)) {

                URI[] patternsURIs = Job.getInstance(conf).getCacheFiles();
                Path patternsPath = new Path(patternsURIs[0].getPath());
                String patternsFileName = patternsPath.getName().toString();
                parseSkipFile(patternsFileName);
                Path punctuationsPath = new Path(patternsURIs[1].getPath());
                String punctuationsFileName = punctuationsPath.getName().toString();
                parseSkipPunctuations(punctuationsFileName);
            }
        }

        private void parseSkipFile(String fileName) {
            try {
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) {
                    patternsToSkip.add(pattern);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file "
                        + StringUtils.stringifyException(ioe));
            }
        }

        private void parseSkipPunctuations(String fileName) {
            try {
                fis = new BufferedReader(new FileReader(fileName));
                String pattern = null;
                while ((pattern = fis.readLine()) != null) {
                    punctuations.add(pattern);
                }
            } catch (IOException ioe) {
                System.err.println("Caught exception while parsing the cached file "
                        + StringUtils.stringifyException(ioe));
            }
        }

        @Override
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            split = (FileSplit) context.getInputSplit();
            String Name = split.getPath().getName();
            int splitIndex = split.getPath().toString().indexOf("file");
            StringTokenizer itr = new StringTokenizer(value.toString());
            String line = value.toString().toLowerCase();
            for (String pattern : punctuations) {
                line = line.replaceAll(pattern, " ");
            }
            while (itr.hasMoreTokens()) {
                String one_word = itr.nextToken();

                if(one_word.length()<3) {
                    continue;
                }
                if(Pattern.compile("^[-\\+]?[\\d]*$").matcher(one_word).matches()) {
                    continue;
                }
                if(patternsToSkip.contains(one_word)){
                    continue;
                }
                keyInfo.set(one_word+"#"+Name);
                context.write(keyInfo, one);
                Counter counter = context.getCounter(
                        CountersEnum.class.getName(),
                        CountersEnum.INPUT_WORDS.toString());
                counter.increment(1);
            }
        }
    }

    public static class Combine extends Reducer<Text, Text, Text, Text> {
        private Text info = new Text();
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (Text value : values) {
                sum += Integer.parseInt(value.toString());
            }
            int splitIndex = key.toString().indexOf(":");
            info.set(key.toString().substring(splitIndex + 1) + ":" + sum);
            key.set(key.toString().substring(0, splitIndex));
            context.write(key, info);
        }
    }



    public static class NewPartitioner extends HashPartitioner<Text, IntWritable> {
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String term = new String();
            term = key.toString().split("#")[0]; // <term#docid>=>term
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    public static class InvertedIndexReducer extends Reducer<Text, IntWritable, Text, NullWritable>{
        private Text result = new Text();
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            String fileList = new String();
            for (Text value : values) {
                fileList += value.toString() + ";";
            }
            result.set(fileList);
        }
        private Text word1 = new Text();
        String temp = new String();
        static Text CurrentItem = new Text(" ");
        static List<String> postingList = new ArrayList<String>();

        public void reduce(Text key, Iterable<Text> values, Context contex)
                throws IOException, InterruptedException {
            int sum = 0;
            for (Text value : values) {
                sum += Integer.parseInt(value.toString());
            }
            int splitIndex = key.toString().indexOf(":");
            word1.set(key.toString().split("#")[0]); //
            temp = key.toString().split("#")[1]; //
            if (!CurrentItem.equals(word1) && !CurrentItem.equals(" ")) { //CurrentItem!=word1, CurrentItem!=" "
                Collections.sort(postingList,Collections.reverseOrder());
                StringBuilder out = new StringBuilder();
                int len = postingList.size();
                int i = 0;
                int count = 0;
                for (String p : postingList) {
                    String fileList = new String();
                    for (Text value : values) {
                        fileList += value.toString() + ";";
                    }
                    result.set(fileList);
                    context.write(key, result);
                    String docId = p.toString().split("#")[1];
                    String wordCount = p.toString().split("#")[0];
                    count += Integer.parseInt(wordCount);
                    out.append(docId + "#" + wordCount);
                    if(i != len-1){
                        out.append(", ");
                        i++;
                    }
                }
                if(count != 0)
                    Text.write(new Text(CurrentItem+": "+out.toString()), NullWritable.get());
                postingList = new ArrayList<String>();
            }
            CurrentItem = new Text(word1);
            postingList.add(word1.toString());
        }

        public void cleanup(Context context) throws IOException,
                InterruptedException {
            Collections.sort(postingList,Collections.reverseOrder());
            StringBuilder out = new StringBuilder();
            int len = postingList.size();
            int i = 0;
            int count=0;
            for (String p : postingList) {
                String docId = p.toString().split("#")[1];
                String wordCount = p.toString().split("#")[0];
                count += Integer.parseInt(wordCount);
                out.append(docId + "#" + wordCount);
                if(i != len-1){
                    out.append(", ");
                    i++;
                }
            }
            if(count != 0)
                context.write(new Text(CurrentItem+": "+out.toString()), NullWritable.get());
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);

        String[] remainingArgs = optionParser.getRemainingArgs();

        for(int i=0;i<remainingArgs.length;i++){
            System.out.println(remainingArgs[i]);
        }
        if (remainingArgs.length != 5) {
            System.err.println("Usage: wordcount <in> <out>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "Inverted Index");
        job.setJarByClass(InvertedIndex.class);
        job.setMapperClass(InvertedIndexMapper.class);
        job.setCombinerClass(Combine.class);
        job.setReducerClass(InvertedIndexReducer.class);
        job.setPartitionerClass(NewPartitioner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        String[] ioArgs = new String[] { "index_in", "index_out" };
        String[] otherArgs = new GenericOptionsParser(conf, ioArgs).getRemainingArgs();
        for (int i = 0; i < remainingArgs.length; ++i) {
            if ("-skip".equals(remainingArgs[i])) {
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.addCacheFile(new Path(remainingArgs[++i]).toUri());
                job.getConfiguration().setBoolean("wordcount.skip.patterns", true);
            } else {
                otherArgs.add(remainingArgs[i]);
            }
        }

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}