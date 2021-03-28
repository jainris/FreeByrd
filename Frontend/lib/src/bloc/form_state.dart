part of 'form_bloc.dart';

class FormTState {
  final String outputDir;
  final String clusteringStrategy;
  final double threshold;
  final int noiseReduction;
  final String noiseFile;
  final List<String> inputFiles;
  final String outputFile;

  FormTState(this.outputDir, this.clusteringStrategy, this.threshold,
      this.noiseReduction, this.noiseFile, this.inputFiles, this.outputFile);

  FormTState copyWith({
    String outputDir,
    String clusteringStrategy,
    double threshold,
    int noiseReduction,
    String noiseFile,
    List<String> inputFiles,
    String outputFile,
  }) {
    return FormTState(
        outputDir ?? this.outputDir,
        clusteringStrategy ?? this.clusteringStrategy,
        threshold ?? this.threshold,
        noiseReduction ?? this.noiseReduction,
        noiseFile ?? this.noiseFile,
        inputFiles ?? this.inputFiles,
        outputFile ?? this.outputFile);
  }

  List<String> convToArgs() {
    List<String> args = [];
    args.add(this.outputFile);
    args.add(this.outputDir);
    if (this.clusteringStrategy == "Dominant Set Clustering")
      args.add('0');
    else {
      args.add('1');
    }
    args.add(this.threshold.toString());
    args.add(this.noiseReduction.toString());
    if (this.noiseReduction == 1) {
      args.add(this.noiseFile.toString());
    }
    args.addAll(this.inputFiles);
    return args;
  }

  // List<String> convToArgs() {
  //   List<String> args = [];
  //   args.add("\"" + this.outputFile + "\"");
  //   args.add("\"" + this.outputDir + "\"");
  //   if (this.clusteringStrategy == "Dominant Set Clustering")
  //     args.add('0');
  //   else {
  //     args.add('1');
  //   }
  //   args.add(this.threshold.toString());
  //   args.add(this.noiseReduction.toString());
  //   if (this.noiseReduction == 1) {
  //     args.add("\"" + this.noiseFile.toString() + "\"");
  //   }
  //   for (String inputFile in this.inputFiles) {
  //     args.add("\'" + inputFile + "\'");
  //   }
  //   // args.addAll(this.inputFiles);
  //   return args;
  // }

  // List<String> convToArgs() {
  //   List<String> args = [];
  //   args.add('"' + this.outputFile + '"');
  //   args.add('"' + this.outputDir + '"');
  //   if (this.clusteringStrategy == "Dominant Set Clustering")
  //     args.add('0');
  //   else {
  //     args.add('1');
  //   }
  //   args.add(this.threshold.toString());
  //   args.add(this.noiseReduction.toString());
  //   if (this.noiseReduction == 1) {
  //     args.add('"' + this.noiseFile.toString() + '"');
  //   }
  //   for (String inputFile in this.inputFiles) {
  //     args.add('"' + inputFile + '"');
  //   }
  //   // args.addAll(this.inputFiles);
  //   return args;
  // }
}
