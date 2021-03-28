import 'package:app12/main.dart';
import 'package:flutter/material.dart';

import 'form.dart';
import 'bloc_2/main_bloc.dart';
import 'package:app12/src/loaded.dart';

class PageGen extends StatelessWidget {
  final i;

  const PageGen(this.i);

  @override
  Widget build(BuildContext context) {
    if (i is Initial) {
      return FormMain();
    } else if (i is Loading) {
      return LoadingPage();
    } else if (i is Loaded) {
      return LoadedPage(i.data, i.outputFile);
    }
  }
}

class LoadingPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: TitleWidg(),
      ),
      body: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            CircularProgressIndicator(
              backgroundColor: Colors.white,
            ),
            Text(
              'Loading',
              style: Theme.of(context).textTheme.subtitle1,
            ),
          ],
        ),
      ),
    );
  }
}
