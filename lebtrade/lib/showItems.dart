import 'package:flutter/material.dart';
import 'package:lebtrade/addItem.dart';
import 'package:lebtrade/homepage.dart';
import 'package:lebtrade/inbox.dart';
import 'package:lebtrade/itemList.dart';
import 'package:lebtrade/models/item.dart';
import 'package:lebtrade/models/user.dart';
import 'package:lebtrade/services/auth.dart';
import 'package:lebtrade/services/database.dart';
import 'package:lebtrade/showItemsCategory.dart';
import 'package:provider/provider.dart';

class ShowItemsHome extends StatefulWidget {
  User loggedIn;
  ShowItemsHome({this.loggedIn});
  @override
  _ShowItemsHomeState createState() => _ShowItemsHomeState();
}

class _ShowItemsHomeState extends State<ShowItemsHome> {
  final AuthService _auth = AuthService();
  String name;

  @override
  Widget build(BuildContext context) {
    DatabaseService().infoCollection.document(widget.loggedIn.uid).get().then((value) async{
      setState(() {
        name = value.data["firstName"] + " " + value.data["lastName"];
      });
    });

    return StreamProvider<List<Item>>.value(
      value: DatabaseService().items,
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: Color.fromRGBO(234, 233, 226, 100),
          title: Text("Browse Items", style: TextStyle(color: Colors.black),),
          centerTitle: true,
          actions: <Widget>[
            FlatButton.icon(
              icon: Icon(Icons.person),
              label: Text('logout'),
              onPressed: () async {
                await _auth.signOut();
              },
            ),
          ],
        ),
        body: SingleChildScrollView(
          child: Column(
            children: [
              Container(
                height: 40,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    FlatButton.icon(onPressed: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ShowItemsCategory(loggedIn: widget.loggedIn,category: "Electronics",),
                          ));
                    }, icon: Icon(Icons.adb_rounded), label: Text("Electronics")),
                    FlatButton.icon(onPressed: (){
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ShowItemsCategory(loggedIn: widget.loggedIn,category: "Vehicles",),
                          ));
                    }, icon: Icon(Icons.accessibility), label: Text("Vehicles")),
                    FlatButton.icon(onPressed: (){
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ShowItemsCategory(loggedIn: widget.loggedIn,category: "Fashion",),
                          ));
                    }, icon: Icon(Icons.airport_shuttle), label: Text("Fashion")),
                    FlatButton.icon(onPressed: (){
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ShowItemsCategory(loggedIn: widget.loggedIn,category: "Home & Games",),
                          ));
                    }, icon: Icon(Icons.android), label: Text("Home & Games"))
                  ],
                ),
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  RaisedButton.icon(onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => AddItem(loggedIn: widget.loggedIn,),
                        ));
                  }, icon: Icon(Icons.transit_enterexit), label: Text("Add an Item")),
                  SizedBox(width: 10,),
                  RaisedButton.icon(onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => HomePage(usrid: widget.loggedIn.uid,),
                        ));
                  }, icon: Icon(Icons.library_books_sharp), label: Text("Wish List")),
                  SizedBox(width: 10,),
                  RaisedButton.icon(onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => Inbox(receiver: name,),
                        ));
                  }, icon: Icon(Icons.inbox), label: Text("Inbox")),
                ],
              ),
              Text("Logged in as $name",textAlign: TextAlign.end,style: TextStyle(fontSize: 14),),
              SizedBox(height: 15,),
              Container(child: ItemList(loggedIn: widget.loggedIn.uid,trade: "",)),
            ],
          ),
        ),
      ),
    );

  }
}
