import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import App from "./App";
import Overview from "./pages/Overview";
import Cache from "./pages/Cache";
import Experiments from "./pages/Experiments";
import ModelsPanel from "./components/ModelsPanel"; // 
import "./index.css";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Overview /> },
      { path: "cache", element: <Cache /> },
      { path: "experiments", element: <Experiments /> },
      { path: "models", element: <ModelsPanel /> }, // 
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
