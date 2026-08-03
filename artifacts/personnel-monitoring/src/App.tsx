import { Switch, Route, Router as WouterRouter, useLocation } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider, useAuth } from "./hooks/use-auth";
import { DepartmentProvider } from "./contexts/department-context";

import Login from "./pages/login";
import Dashboard from "./pages/dashboard";
import OfficePage from "./pages/office";
import StaffMonitoring from "./pages/staff-monitoring";
import Register from "./pages/register";
import Accounts from "./pages/accounts";
import SamplePDFs from "./pages/sample-pdfs";
import NotFound from "./pages/not-found";

const queryClient = new QueryClient();

function ProtectedRoute({
  component: Component,
  allowedRoles,
  ...rest
}: {
  component: React.ComponentType<any>;
  allowedRoles?: string[];
  path?: string;
  [key: string]: any;
}) {
  return (
    <Route {...rest}>
      {(params) => {
        const { user, isLoading } = useAuth();
        const [, setLocation] = useLocation();

        if (isLoading) {
          return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
              <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
            </div>
          );
        }

        if (!user) {
          setLocation("/login");
          return null;
        }

        if (allowedRoles && !allowedRoles.includes(user.role)) {
          setLocation(user.role === "admin" ? "/dashboard" : "/staff-monitoring");
          return null;
        }

        return <Component {...params} />;
      }}
    </Route>
  );
}

function Router() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <Switch>
      <Route path="/">
        {() => {
          if (user) {
            window.location.href =
              import.meta.env.BASE_URL + (user.role === "admin" ? "dashboard" : "staff-monitoring");
          } else {
            window.location.href = import.meta.env.BASE_URL + "login";
          }
          return null;
        }}
      </Route>
      <Route path="/login" component={Login} />

      <ProtectedRoute path="/dashboard" component={Dashboard} allowedRoles={["admin"]} />
      <ProtectedRoute path="/office/:slug" component={OfficePage} allowedRoles={["admin"]} />
      <ProtectedRoute path="/staff-monitoring" component={StaffMonitoring} allowedRoles={["admin", "user"]} />
      <ProtectedRoute path="/register" component={Register} allowedRoles={["admin"]} />
      <ProtectedRoute path="/register/:id" component={Register} allowedRoles={["admin"]} />
      <ProtectedRoute path="/accounts" component={Accounts} allowedRoles={["admin"]} />
      <ProtectedRoute path="/sample-pdfs" component={SamplePDFs} allowedRoles={["admin"]} />

      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
        <AuthProvider>
          <DepartmentProvider>
            <Router />
          </DepartmentProvider>
        </AuthProvider>
      </WouterRouter>
    </QueryClientProvider>
  );
}

export default App;
