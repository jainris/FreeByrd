import 'package:freebyrd/main.dart';
import 'package:freebyrd/src/bloc/form_bloc.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:filepicker_windows/filepicker_windows.dart';

import 'bloc_2/main_bloc.dart';

class FormMain extends StatelessWidget {
  final double width = 400;
  @override
  Widget build(BuildContext context) {
    return BlocBuilder<FormBloc, FormTState>(builder: (context, state) {
      TextStyle fileNameStyle =
          Theme.of(context).textTheme.bodyText1.apply(color: Colors.grey);

      List<Widget> inputFiles = [];
      List<Widget> saveFile = [];
      List<Widget> outDir = [];
      List<Widget> noiseFile = [];
      for (String inputFile in state.inputFiles) {
        inputFiles.add(Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            Container(
              width: width * 0.8,
              child: Text(
                inputFile,
                style: fileNameStyle,
              ),
            ),
            IconButton(
              icon: Icon(Icons.cancel_outlined),
              onPressed: (() {
                context.read<FormBloc>().add(FormInputFileRemoved(inputFile));
              }),
            )
          ],
        ));
      }
      if (state.outputFile != "") {
        saveFile = [
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Container(
                width: width * 0.8,
                child: Text(
                  state.outputFile,
                  style: fileNameStyle,
                ),
              ),
              IconButton(
                icon: Icon(Icons.cancel_outlined),
                onPressed: (() {
                  context.read<FormBloc>().add(FormOutputFileRemoved());
                }),
              )
            ],
          )
        ];
      }
      if (state.outputDir != "") {
        outDir = [
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Container(
                width: width * 0.8,
                child: Text(
                  state.outputDir,
                  style: fileNameStyle,
                ),
              ),
              IconButton(
                icon: Icon(Icons.cancel_outlined),
                onPressed: (() {
                  context.read<FormBloc>().add(FormOutputDirRemoved());
                }),
              )
            ],
          )
        ];
      }
      if (state.noiseFile != "") {
        noiseFile = [
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Container(
                width: width * 0.8,
                child: Text(
                  state.noiseFile,
                  style: fileNameStyle,
                ),
              ),
              IconButton(
                icon: Icon(Icons.cancel_outlined),
                onPressed: (() {
                  context.read<FormBloc>().add(FormNRChanged(0));
                  context.read<FormBloc>().add(FormNoiseFileRemoved());
                }),
              )
            ],
          )
        ];
      }
      return Scaffold(
        appBar: AppBar(
          title: TitleWidg(),
        ),
        body: Form(
          key: Key('123'),
          child: Scrollbar(
            child: Align(
              alignment: Alignment.topCenter,
              child: Card(
                child: SingleChildScrollView(
                  padding: EdgeInsets.all(16),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: width),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        ...[
                          Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Input Files: ',
                                      style:
                                          Theme.of(context).textTheme.bodyText1,
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  TextButton(
                                    child: Text('Browse'),
                                    onPressed: () async {
                                      // TODO: Look into multiple files
                                      var file = OpenFilePicker()
                                        ..filterSpecification = {
                                          'Audio File (*.wav)': '*.wav',
                                        }
                                        ..defaultFilterIndex = 0
                                        ..defaultExtension = 'wav'
                                        ..title = 'Select an audio file';

                                      try {
                                        var inputFile = file.getFile().path;
                                        context
                                            .read<FormBloc>()
                                            .add(FormInputFileAdded(inputFile));
                                      } catch (e) {}
                                    },
                                  ),
                                ],
                              ),
                              ...inputFiles,
                            ],
                          ),
                          Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Save Output File as: ',
                                      style:
                                          Theme.of(context).textTheme.bodyText1,
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  TextButton(
                                    child: Text('Save as'),
                                    onPressed: () async {
                                      var file = SaveFilePicker()
                                        ..filterSpecification = {
                                          'CSV File (*.csv)': '*.csv',
                                        }
                                        ..defaultFilterIndex = 0
                                        ..defaultExtension = 'csv'
                                        ..title = 'Save the CSV file as';

                                      try {
                                        var inputFile = file.getFile().path;
                                        context.read<FormBloc>().add(
                                            FormOutputFileAdded(inputFile));
                                      } catch (e) {}
                                    },
                                  ),
                                ],
                              ),
                              ...saveFile,
                            ],
                          ),
                          Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Output Folder (for saving the samples): ',
                                      style:
                                          Theme.of(context).textTheme.bodyText1,
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  TextButton(
                                    child: Text('Browse'),
                                    onPressed: () async {
                                      var outputDir = DirectoryPicker()
                                        ..title = 'Output Directory';
                                      try {
                                        var outputDirPath =
                                            outputDir.getDirectory().path;
                                        context.read<FormBloc>().add(
                                            FormOutputDirAdded(outputDirPath));
                                      } catch (e) {}
                                    },
                                  ),
                                ],
                              ),
                              ...outDir,
                            ],
                          ),
                          Column(
                            mainAxisAlignment: MainAxisAlignment.start,
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    'Threshold',
                                    style:
                                        Theme.of(context).textTheme.bodyText1,
                                  ),
                                ],
                              ),
                              Text(
                                state.threshold.toStringAsFixed(1),
                                style: Theme.of(context).textTheme.subtitle1,
                              ),
                              Slider(
                                min: 10,
                                max: 90,
                                divisions: 800,
                                value: state.threshold,
                                onChanged: (threshold) => context
                                    .read<FormBloc>()
                                    .add(FormThresholdChanged(threshold)),
                              ),
                            ],
                          ),
                          Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Noise File (optional): ',
                                      style:
                                          Theme.of(context).textTheme.bodyText1,
                                      textAlign: TextAlign.left,
                                    ),
                                  ),
                                  TextButton(
                                    child: Text('Browse'),
                                    onPressed: () async {
                                      var file = OpenFilePicker()
                                        ..filterSpecification = {
                                          'Audio File (*.wav)': '*.wav',
                                        }
                                        ..defaultFilterIndex = 0
                                        ..defaultExtension = 'wav'
                                        ..title = 'Select an audio file';

                                      try {
                                        var inputFile = file.getFile().path;
                                        context
                                            .read<FormBloc>()
                                            .add(FormNRChanged(1));
                                        context
                                            .read<FormBloc>()
                                            .add(FormNoiseFileAdded(inputFile));
                                      } catch (e) {}
                                    },
                                  ),
                                ],
                              ),
                              ...noiseFile,
                            ],
                          ),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Expanded(
                                child: Text(
                                  'Clustering Strategy: ',
                                  style: Theme.of(context).textTheme.bodyText1,
                                  textAlign: TextAlign.left,
                                ),
                              ),
                              DropdownButton<String>(
                                value: state.clusteringStrategy,
                                icon: Icon(Icons.arrow_downward),
                                iconSize: 24,
                                elevation: 16,
                                style: TextStyle(color: Colors.deepPurple),
                                underline: Container(
                                  height: 2,
                                  color: Colors.deepPurpleAccent,
                                ),
                                onChanged: (String newValue) {
                                  context
                                      .read<FormBloc>()
                                      .add(FormStrategyChanged(newValue));
                                },
                                items: <String>[
                                  'Dominant Set Clustering',
                                  'Hybrid SOM and K-Means'
                                ].map<DropdownMenuItem<String>>((String value) {
                                  return DropdownMenuItem<String>(
                                    value: value,
                                    child: Text(value),
                                  );
                                }).toList(),
                              ),
                            ],
                          ),
                          TextButton(
                            child: Text('Run Model'),
                            onPressed: () {
                              context
                                  .read<MainBloc>()
                                  .add(SubmitMainEvent(state));
                            },
                          )
                        ].expand(
                          (widget) => [
                            widget,
                            SizedBox(
                              height: 24,
                            )
                          ],
                        )
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      );
    });
  }
}
