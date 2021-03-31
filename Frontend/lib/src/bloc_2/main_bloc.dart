import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;

import 'package:app12/src/bloc/form_bloc.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class MainBloc extends Bloc<MainEvent, MainState> {
  MainBloc() : super(Initial());

  @override
  Stream<MainState> mapEventToState(MainEvent event) async* {
    if (event is SubmitMainEvent) {
      ProcessResult result;
      yield Loading();
      try {
        var command = Directory.current.parent.path;

        List<String> args = event.state.convToArgs();
        print(args);

        var string = "";
        for (var arg in args) {
          string += arg + " ";
        }

        command = "cd " + command + " & thefiletoberun.bat " + string;
        result = await Process.run(command, [], runInShell: true);

        List<List<String>> data = [];

        String filePath =
            p.join(event.state.outputDir, "flutter_aux_output.csv");

        var input = File(filePath).openRead();
        var fs = await input
            .transform(utf8.decoder)
            .transform(new LineSplitter())
            .listen((String line) {
          List<String> row = line.split(',');
          data.add(row);
          print(data);
          print(row.toList());
        }).asFuture();
        yield Loaded(event.state.outputFile, data);
      } catch (e) {
        yield Error(e, result);
      }
    }
  }
}

abstract class MainEvent {}

class SubmitMainEvent extends MainEvent {
  final FormTState state;

  SubmitMainEvent(this.state);
}

abstract class MainState {}

class Initial extends MainState {}

class Loading extends MainState {}

class Loaded extends MainState {
  final String outputFile;
  final data;

  Loaded(this.outputFile, this.data);
}

class Error extends MainState {
  final Exception e;
  final ProcessResult r;

  Error(this.e, this.r);
}
