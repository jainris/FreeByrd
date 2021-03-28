import 'package:flutter_bloc/flutter_bloc.dart';

part 'form_event.dart';
part 'form_state.dart';

class FormBloc extends Bloc<FormEvent, FormTState> {
  FormBloc()
      : super(FormTState("", "Dominant Set Clustering", 50, 0, "", [], ""));
  @override
  Stream<FormTState> mapEventToState(FormEvent event) async* {
    if (event is FormOutputDirAdded) {
      yield state.copyWith(outputDir: event.outputDir);
    } else if (event is FormOutputDirRemoved) {
      yield state.copyWith(outputDir: "");
    } else if (event is FormStrategyChanged) {
      yield state.copyWith(clusteringStrategy: event.clusteringStrategy);
    } else if (event is FormThresholdChanged) {
      yield state.copyWith(threshold: event.threshold);
    } else if (event is FormNRChanged) {
      yield state.copyWith(noiseReduction: event.nR);
    } else if (event is FormNoiseFileAdded) {
      yield state.copyWith(noiseFile: event.noiseFile);
    } else if (event is FormNoiseFileRemoved) {
      yield state.copyWith(noiseFile: "");
    } else if (event is FormInputFileAdded) {
      List<String> inputFiles = List.from(state.inputFiles);
      inputFiles.add(event.fileName);
      yield state.copyWith(inputFiles: inputFiles);
    } else if (event is FormInputFileRemoved) {
      List<String> inputFiles = List.from(state.inputFiles);
      inputFiles.remove(event.fileName);
      yield state.copyWith(inputFiles: inputFiles);
    } else if (event is FormOutputFileAdded) {
      yield state.copyWith(outputFile: event.outputFile);
    } else if (event is FormOutputFileRemoved) {
      yield state.copyWith(outputFile: "");
    }
  }
}
