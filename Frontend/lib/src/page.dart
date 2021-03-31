import 'dart:io';

import 'package:freebyrd/main.dart';
import 'package:flutter/material.dart';

import 'form.dart';
import 'bloc_2/main_bloc.dart';
import 'package:freebyrd/src/loaded.dart';

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
    } else if (i is Error) {
      return ErrorPage(i.e, i.r);
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
            Container(
              padding: EdgeInsets.all(10),
              child: Text(
                'Loading',
                style: Theme.of(context).textTheme.headline5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ErrorPage extends StatefulWidget {
  final Exception e;
  final ProcessResult r;

  ErrorPage(this.e, this.r);

  @override
  _ErrorPageState createState() => _ErrorPageState();
}

class _ErrorPageState extends State<ErrorPage> {
  bool showLog = false;
  bool moreInfo = false;
  final double width = 400;

  @override
  Widget build(BuildContext context) {
    List<Widget> Log = [];
    if (this.showLog) {
      Log = [
        Container(
          width: this.width,
          child: Text(widget.e.toString()),
        )
      ];
      if (this.moreInfo) {
        Log.addAll(
          [
            Container(
              width: 400,
              child: Text('Exit Code: ' + widget.r.exitCode.toString()),
            ),
            Container(
              width: 400,
              child: Text(
                'Stdout: ',
                style: Theme.of(context).textTheme.headline5,
              ),
            ),
            Container(
              width: 400,
              child: Text(widget.r.stdout),
            ),
            Container(
              width: 400,
              child: Text(
                'Stderr: ',
                style: Theme.of(context).textTheme.headline5,
              ),
            ),
            Container(
              width: 400,
              child: Text(widget.r.stderr),
            ),
          ],
        );
      } else {
        Log.add(
          TextButton(
            child: Text('Show Process Information'),
            onPressed: () {
              setState(() {
                this.moreInfo = true;
              });
            },
          ),
        );
      }
    }
    return Scaffold(
      appBar: AppBar(
        title: TitleWidg(),
      ),
      backgroundColor: Colors.red,
      body: Scrollbar(
        child: Center(
          child: Card(
            child: SingleChildScrollView(
              padding: EdgeInsets.all(16),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Container(
                    width: this.width,
                    alignment: Alignment.center,
                    child: Text(
                      'Error',
                      style: Theme.of(context).textTheme.headline5,
                    ),
                  ),
                  TextButton(
                    child: this.showLog ? Text('Hide Log') : Text('Show Log'),
                    onPressed: () {
                      setState(() {
                        this.showLog = !this.showLog;
                        this.moreInfo = false;
                      });
                    },
                  ),
                  ...Log,
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
