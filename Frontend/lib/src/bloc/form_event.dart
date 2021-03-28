part of 'form_bloc.dart';

abstract class FormEvent {
  const FormEvent();
}

class FormOutputDirAdded extends FormEvent {
  final String outputDir;

  FormOutputDirAdded(this.outputDir);
}

class FormOutputDirRemoved extends FormEvent {}

class FormStrategyChanged extends FormEvent {
  final String clusteringStrategy;

  FormStrategyChanged(this.clusteringStrategy);
}

class FormThresholdChanged extends FormEvent {
  final double threshold;

  FormThresholdChanged(this.threshold);
}

class FormNRChanged extends FormEvent {
  final int nR;

  FormNRChanged(this.nR);
}

class FormNoiseFileAdded extends FormEvent {
  final String noiseFile;

  FormNoiseFileAdded(this.noiseFile);
}

class FormNoiseFileRemoved extends FormEvent {}

class FormInputFileAdded extends FormEvent {
  final String fileName;

  FormInputFileAdded(this.fileName);
}

class FormInputFileRemoved extends FormEvent {
  final String fileName;

  FormInputFileRemoved(this.fileName);
}

class FormOutputFileAdded extends FormEvent {
  final String outputFile;

  FormOutputFileAdded(this.outputFile);
}

class FormOutputFileRemoved extends FormEvent {}
