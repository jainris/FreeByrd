import 'package:app12/src/bloc/form_bloc.dart';
import 'package:app12/src/bloc_2/main_bloc.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'src/page.dart';
import 'src/bloc_2/main_bloc.dart';

/// Custom [BlocObserver] which observes all bloc and cubit instances.
class SimpleBlocObserver extends BlocObserver {
  @override
  void onEvent(Bloc bloc, Object event) {
    print(event);
    super.onEvent(bloc, event);
  }

  @override
  void onChange(Cubit cubit, Change change) {
    print(change);
    super.onChange(cubit, change);
  }

  @override
  void onTransition(Bloc bloc, Transition transition) {
    print(transition);
    super.onTransition(bloc, transition);
  }

  @override
  void onError(Cubit cubit, Object error, StackTrace stackTrace) {
    print(error);
    super.onError(cubit, error, stackTrace);
  }
}

void main() {
  Bloc.observer = SimpleBlocObserver();
  runApp(MainApp());
}

class MainApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BlocProvider(
        create: (_) => MainBloc(),
        child: BlocBuilder<MainBloc, MainState>(
          builder: (buildContext, state) {
            return BlocProvider(
              create: (_) => FormBloc(),
              child: MaterialApp(
                title: 'Byrd Detect',
                theme: ThemeData(primarySwatch: Colors.teal),
                home: PageGen(state),
              ),
            );
          },
        ));
  }
}

class TitleWidg extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(8),
      child: Text('Byrd Detect'),
    );
  }
}
